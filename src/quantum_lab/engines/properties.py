"""Reference-density populations and molecular harmonic analysis.

Normal modes are mass-normalized Cartesian displacements, not trajectories.
Thermochemistry uses an ideal gas, rigid rotor and unscaled harmonic oscillator;
it is withheld for a nonstationary structure or nonpositive vibrational modes.
"""
from __future__ import annotations

import csv
import inspect
from copy import copy
from pathlib import Path

import numpy as np
from pyscf import lib
from pyscf.data import nist
from pyscf.hessian import thermo
from scipy.constants import Avogadro, atomic_mass, c, elementary_charge, epsilon_0

from ..storage import atomic_json

STATIONARITY_THRESHOLD = 4.5e-4  # maximum Cartesian gradient, Eh/bohr
IR_STEP_BOHR = 0.005
# d(mu/e bohr)/d(r/bohr) contracted with modes normalized in amu^-1/2.
# SI integrated molar intensity, converted from m/mol to km/mol.
IR_INTENSITY_FACTOR = Avogadro * elementary_charge**2 / (12 * epsilon_0 * c**2 * atomic_mass) / 1000
CM1_TO_HARTREE = nist.PLANCK * nist.LIGHT_SPEED_SI * 100 / nist.HARTREE2J


def _finite_array(value, description: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite {description}; no successful property result was produced.")
    return array


def analyze_properties(mf) -> dict:
    """Analyze the supplied converged mean-field density without new SCF work.

    Mulliken and symmetric Löwdin populations depend on the selected AO basis.
    When the caller subsequently computes a correlated energy, these continue
    to describe the HF reference and must retain that explicit model label.
    """
    if not mf.converged:
        raise ValueError("Population analysis requires a converged SCF reference.")
    mol = mf.mol
    dm = _finite_array(mf.make_rdm1(), "density matrix")
    unrestricted = dm.ndim == 3
    total_dm = dm.sum(axis=0) if unrestricted else dm
    overlap = _finite_array(mf.get_ovlp(), "overlap matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if np.min(eigenvalues) <= 0:
        raise ValueError("Löwdin populations require a positive-definite overlap matrix.")
    root_overlap = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    mulliken_ao = np.diag(total_dm @ overlap)
    lowdin_ao = np.diag(root_overlap @ total_dm @ root_overlap)
    slices = mol.aoslice_by_atom()
    nuclear = mol.atom_charges()
    mulliken = [float(z - mulliken_ao[p0:p1].sum()) for z, (_, _, p0, p1) in zip(nuclear, slices)]
    lowdin = [float(z - lowdin_ao[p0:p1].sum()) for z, (_, _, p0, p1) in zip(nuclear, slices)]
    dipole = _finite_array(mf.dip_moment(unit="Debye", verbose=0), "dipole moment")
    spin_square, multiplicity = mf.spin_square()
    is_dft = hasattr(mf, "xc")
    model = ("UKS" if unrestricted else "RKS") if is_dft else ("UHF" if unrestricted else "RHF")
    warnings = ["Mulliken·Löwdin 부분 전하는 기저와 분할 방식에 의존하며 직접 관측되는 원자 전하가 아닙니다."]
    if mol.charge:
        warnings.append("전하를 띤 분자의 쌍극자 모멘트는 좌표 원점에 의존합니다.")
    if getattr(mol, "_ecp", None):
        warnings.append("ECP 계산의 전자밀도와 population은 명시적으로 남긴 가전자와 유효 핵전하를 사용합니다.")
    return {
        "reference_model": model,
        "mulliken_charges": mulliken,
        "lowdin_charges": lowdin,
        "mulliken_charge_sum": float(sum(mulliken)),
        "lowdin_charge_sum": float(sum(lowdin)),
        "dipole_debye": dipole.tolist(),
        "dipole_magnitude_debye": float(np.linalg.norm(dipole)),
        "spin_square": float(spin_square),
        "spin_multiplicity": float(multiplicity),
        "expected_spin_square": float(mol.spin / 2 * (mol.spin / 2 + 1)),
        "units": {"charges": "elementary_charge", "dipole": "debye"},
        "warnings": warnings,
    }


def _displaced_reference(mf, mol):
    """Retain numerical settings while giving the displaced SCF separate caches."""
    displaced = mf.copy()
    # SCF.copy is shallow. Resetting shared DFT grids would otherwise mutate the
    # parent's grids and associate its density with the final displaced geometry.
    for name in ("grids", "nlcgrids", "second_grids"):
        grid = getattr(mf, name, None)
        if grid is not None:
            setattr(displaced, name, copy(grid))
    displaced.reset(mol)
    displaced.chkfile = None
    displaced.callback = None
    displaced.verbose = 0
    return displaced


def _checked_hessian(solver, workdir: Path) -> tuple[np.ndarray, list[dict]]:
    """Verify every returned CPSCF Krylov solution against its actual equation.

    Some Krylov paths warn and return when their subspace becomes dependent
    before the residual meets tolerance. A finite Hessian alone cannot certify
    that response solution. Calls are scoped to this calculation process and
    the original library function is restored even when a solver raises.
    """
    original = lib.krylov
    signature = inspect.signature(original)
    checks = []

    def checked(aop, b, *args, **kwargs):
        bound = signature.bind(aop, b, *args, **kwargs)
        bound.apply_defaults()
        tolerance = float(bound.arguments["tol"])
        result = original(aop, b, *args, **kwargs)
        target = np.asarray(b)
        solution = np.asarray(result)
        image = np.asarray(aop(solution)).reshape(target.shape)
        residual = target - solution.reshape(target.shape) - image
        norms = np.linalg.norm(residual.reshape(-1, target.shape[-1]), axis=1)
        maximum = float(np.max(norms))
        check = {"requested_tolerance": tolerance, "max_residual_norm": maximum,
                 "right_hand_sides": int(len(norms)), "max_cycle": int(bound.arguments["max_cycle"])}
        checks.append(check)
        if not np.isfinite(maximum) or maximum > tolerance:
            raise RuntimeError(f"CPSCF response residual {maximum:.6g} exceeds requested tolerance {tolerance:.6g}.")
        return result

    try:
        with lib.temporary_env(lib, krylov=checked):
            hessian = _finite_array(solver.kernel(), "Hessian")
    except Exception as exc:
        atomic_json(workdir / "hessian-response.json", {"completed": False, "checks": checks, "error": str(exc)})
        raise RuntimeError(f"Analytic Hessian response failed; no frequencies were accepted: {exc}") from exc
    atomic_json(workdir / "hessian-response.json", {"completed": True, "checks": checks})
    return hessian, checks


def _ir_intensities(mf, modes: np.ndarray, emit, workdir: Path) -> tuple[np.ndarray | None, dict]:
    coordinates = mf.mol.atom_coords(unit="Bohr")
    derivatives = np.zeros((mf.mol.natm, 3, 3))
    records = []
    initial_density = np.asarray(mf.make_rdm1())
    total = mf.mol.natm * 6
    metadata = {
        "requested": True, "computed": False, "finite_difference_step_bohr": IR_STEP_BOHR,
        "displaced_calculations": 0,
        "method": "central finite difference of total SCF dipole; 6N displacements",
        "intensity_factor_km_mol": IR_INTENSITY_FACTOR,
    }
    for atom in range(mf.mol.natm):
        for axis in range(3):
            dipoles = []
            for sign in (-1, 1):
                positions = coordinates.copy()
                positions[atom, axis] += sign * IR_STEP_BOHR
                mol = mf.mol.copy()
                mol.set_geom_(positions, unit="Bohr")
                displaced = _displaced_reference(mf, mol)
                emit("stage", {"name": "ir_displacement", "completed": len(records), "total": total,
                               "atom": atom, "axis": axis, "sign": sign,
                               "message": "Computing a displaced SCF dipole for IR intensities."})
                displaced.kernel(dm0=initial_density)
                record = {"atom": atom, "axis": axis, "displacement_bohr": sign * IR_STEP_BOHR,
                          "scf_converged": bool(displaced.converged),
                          "energy_hartree": float(displaced.e_tot)}
                records.append(record)
                metadata["displaced_calculations"] = len(records)
                if not displaced.converged or not np.isfinite(displaced.e_tot):
                    metadata["error"] = "A displaced SCF did not converge; IR intensities are unavailable."
                    atomic_json(workdir / "ir-displacements.json", {"settings": metadata, "calculations": records})
                    return None, metadata
                dipole = _finite_array(displaced.dip_moment(unit="AU", verbose=0), "displaced dipole")
                record["dipole_atomic_units"] = dipole.tolist()
                dipoles.append(dipole)
                atomic_json(workdir / "ir-displacements.json", {"settings": metadata, "calculations": records})
            derivatives[atom, axis] = (dipoles[1] - dipoles[0]) / (2 * IR_STEP_BOHR)
    # derivatives[atom, coordinate, dipole_component]; modes[mode, atom, coordinate].
    normal_derivatives = np.einsum("acp,mac->mp", derivatives, modes)
    intensities = IR_INTENSITY_FACTOR * np.einsum("mp,mp->m", normal_derivatives, normal_derivatives)
    metadata["computed"] = True
    atomic_json(workdir / "ir-displacements.json", {
        "settings": metadata, "calculations": records,
        "dipole_derivatives_e": derivatives.tolist(),
        "normal_dipole_derivatives_e_per_sqrt_amu": normal_derivatives.tolist(),
    })
    emit("stage", {"name": "ir_displacement", "completed": total, "total": total,
                   "message": "All displaced SCF dipoles converged."})
    return intensities, metadata


def run_frequencies(mf, spec: dict, emit, workdir: Path) -> dict:
    """Compute an analytic Hessian, projected modes, optional IR and valid RRHO data."""
    if not mf.converged:
        raise ValueError("Frequencies require a converged SCF reference.")
    if mf.mol.spin or not spec.get("restricted", True) or getattr(mf.mol, "_ecp", None):
        raise ValueError("Frequency analysis supports closed-shell RHF/RKS without ECP.")
    if spec.get("method", "HF") not in {"HF", "DFT"}:
        raise ValueError("Analytic frequencies support HF and DFT references, not correlated energies.")
    if hasattr(mf, "xc") and str(mf.xc).upper() not in {"PBE", "PBE0", "B3LYP"}:
        raise ValueError("The frequency path supports PBE, PBE0 and B3LYP.")
    temperature, pressure = float(spec.get("temperature_kelvin", 298.15)), float(spec.get("pressure_pa", 101325))
    if not np.isfinite([temperature, pressure]).all() or temperature <= 0 or pressure <= 0:
        raise ValueError("Thermochemistry temperature and pressure must be finite and positive.")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    emit("stage", {"name": "hessian", "message": "Computing the analytic Cartesian Hessian."})
    hessian_solver = mf.Hessian()
    hessian_solver.max_cycle = int(spec.get("max_cycle", mf.max_cycle))
    hessian_solver.base = mf.copy()
    hessian_solver.base.conv_tol_cpscf = min(float(mf.conv_tol_cpscf), float(spec.get("conv_tol", mf.conv_tol)))
    hessian, response_checks = _checked_hessian(hessian_solver, workdir)
    gradient = _finite_array(mf.nuc_grad_method().kernel(), "nuclear gradient")
    gradient_max = float(np.max(np.abs(gradient)))
    stationary = gradient_max <= STATIONARITY_THRESHOLD
    masses = _finite_array(mf.mol.atom_mass_list(isotope_avg=True), "atomic masses")
    harmonic = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False, mass=masses)
    frequencies = _finite_array(harmonic["freq_wavenumber"], "harmonic frequencies")
    modes = _finite_array(harmonic["norm_mode"], "normal modes")
    reduced_masses = _finite_array(harmonic["reduced_mass"], "reduced masses")
    imaginary_count = int(np.sum(frequencies < 0))
    positive_modes = bool(np.all(frequencies > 0))
    warnings = [
        "조화 진동은 입력 구조에서의 국소 곡률입니다. 모드 애니메이션은 임의 진폭의 모식도이며 실제 시간 동역학이 아닙니다.",
        "동위원소 평균 원자 질량을 사용하며 주파수 경험적 보정 계수는 1입니다.",
    ]
    if not stationary:
        warnings.append("입력 구조의 최대 힘이 정지점 기준을 넘습니다. 먼저 구조를 최적화하세요. 열화학 값은 제공하지 않습니다.")
        warnings.append("양의 모드로 구한 ZPE 합도 이 비정지 구조의 국소 조화 값이며, 평형 구조의 영점 에너지로 확정하지 않습니다.")
    if not positive_modes:
        warnings.append("0 이하의 진동 모드가 있습니다. 허수 진동은 음수 cm⁻¹로 표시하며 ZPE·열화학 합산에서 임의로 제외하지 않습니다.")
    if np.any((frequencies > 0) & (frequencies < 100)):
        warnings.append("낮은 진동수의 조화 진동자 엔트로피는 민감합니다. RRHO는 내부 회전·비조화성을 다루지 않습니다.")
    zero_point = float(0.5 * frequencies.sum() * CM1_TO_HARTREE) if positive_modes else None
    thermochemistry = None
    if stationary and positive_modes:
        values = thermo.thermo(mf, harmonic["freq_au"], temperature=temperature, pressure=pressure)
        thermochemistry = {
            "model": "ideal-gas RRHO", "temperature_kelvin": temperature, "pressure_pa": pressure,
            "internal_energy_hartree": float(values["E_tot"][0]),
            "enthalpy_hartree": float(values["H_tot"][0]),
            "gibbs_energy_hartree": float(values["G_tot"][0]),
            "entropy_hartree_per_kelvin": float(values["S_tot"][0]),
            "heat_capacity_cv_hartree_per_kelvin": float(values["Cv_tot"][0]),
            "heat_capacity_cp_hartree_per_kelvin": float(values["Cp_tot"][0]),
            "rotational_symmetry_number": int(values["sym_number"][0]),
            "frequency_scaling_factor": 1,
        }
        warnings.append("열화학은 지정한 온도·압력의 이상 기체 RRHO 근사입니다. 용액·고체·비조화 효과는 포함하지 않습니다.")
    intensities = None
    ir = {"requested": bool(spec.get("ir_intensities", False)), "computed": False,
          "finite_difference_step_bohr": IR_STEP_BOHR, "displaced_calculations": 0}
    if ir["requested"]:
        intensities, ir = _ir_intensities(mf, modes, emit, workdir)
        if not ir["computed"]:
            warnings.append("IR 유한 차분 중 수렴하지 않은 SCF가 있어 세기를 제공하지 않습니다. 진동 결과와 부분 기록은 보존합니다.")
        else:
            warnings.append("IR 세기는 ±0.005 bohr 중앙 유한 차분으로 계산했습니다. 격자·SCF·변위 크기 오차의 별도 수렴 검사가 필요합니다.")
    artifacts = ["vibrations.json", "vibrations.csv", "vibrations.npz", "hessian-response.json"]
    if ir["requested"]:
        artifacts.append("ir-displacements.json")
    result = {
        "converged": not ir["requested"] or ir["computed"],
        "frequencies_cm1": frequencies.tolist(),
        "modes": [{"index": index, "frequency_cm1": float(freq), "imaginary": bool(freq < 0),
                   "reduced_mass_amu": float(reduced_masses[index]), "displacements": modes[index].tolist(),
                   "ir_intensity_km_mol": float(intensities[index]) if intensities is not None and freq > 0 else None}
                  for index, freq in enumerate(frequencies)],
        "masses_amu": masses.tolist(),
        "gradient_max_hartree_per_bohr": gradient_max,
        "stationary": stationary,
        "stationarity_threshold_hartree_per_bohr": STATIONARITY_THRESHOLD,
        "imaginary_modes": imaginary_count,
        "zero_point_energy_hartree": zero_point,
        "thermochemistry": thermochemistry,
        "ir": ir,
        "hessian_method": "analytic RHF/RKS Hessian; translation/rotation projected",
        "response_solver": {"cpscf_tolerance": hessian_solver.base.conv_tol_cpscf,
                            "max_cycle": hessian_solver.max_cycle, "residual_checks": response_checks},
        "mode_normalization": "sum_atom mass_amu * dot(displacement,displacement) = 1",
        "units": {"hessian": "hartree/bohr^2", "frequency": "cm^-1", "mass": "amu",
                  "displacements": "amu^-1/2", "ir_intensity": "km/mol", "energy": "hartree"},
        "constants_providers": {"frequency_and_thermochemistry": "pyscf.data.nist",
                                "ir_intensity": "scipy.constants"},
        "warnings": warnings, "artifacts": artifacts,
    }
    atomic_json(workdir / "vibrations.json", result)
    np.savez_compressed(workdir / "vibrations.npz", hessian_hartree_per_bohr2=hessian,
                        gradient_hartree_per_bohr=gradient, frequencies_cm1=frequencies,
                        modes=modes, masses_amu=masses)
    with (workdir / "vibrations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "frequency_cm1", "imaginary",
                                                   "reduced_mass_amu", "ir_intensity_km_mol"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["modes"])
    emit("stage", {"name": "frequencies", "modes": len(frequencies), "imaginary_modes": imaginary_count,
                   "message": "Harmonic analysis complete; stationarity and thermochemistry checks recorded."})
    return result
