"""A small, inspectable NumPy restricted Hartree--Fock implementation.

PySCF supplies Gaussian atomic-orbital integrals only. Orthogonalization,
diagonalization, Coulomb/exchange assembly, DIIS and the SCF loop below are
implemented here. This is a teaching solver, not an independent integral engine.
The restricted single-determinant model is unsuitable for many stretched bonds.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib
from pyscf.tools import cubegen

MAX_AOS = 80


class _DIIS:
    """Pulay extrapolation of Fock matrices using orthogonal-basis residuals."""

    def __init__(self, size: int = 8):
        self.size = size
        self.focks: list[np.ndarray] = []
        self.errors: list[np.ndarray] = []

    def extrapolate(self, fock: np.ndarray, error: np.ndarray) -> np.ndarray:
        self.focks.append(fock.copy())
        self.errors.append(error.ravel().copy())
        self.focks = self.focks[-self.size :]
        self.errors = self.errors[-self.size :]
        n = len(self.focks)
        if n < 2:
            return fock
        gram = np.array([[np.dot(a, b) for b in self.errors] for a in self.errors])
        scale = max(float(np.max(np.abs(gram))), 1e-30)
        b = np.empty((n + 1, n + 1))
        b[:n, :n] = gram / scale
        b[n, :] = -1.0
        b[:, n] = -1.0
        b[n, n] = 0.0
        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0
        try:
            weights = np.linalg.solve(b, rhs)[:n]
        except np.linalg.LinAlgError:
            # Dependent residuals occur near convergence. A smaller history is
            # preferable to silently accepting a singular extrapolation.
            self.focks.pop(0)
            self.errors.pop(0)
            return fock
        if not np.all(np.isfinite(weights)) or np.max(np.abs(weights)) > 1e6:
            return fock
        return np.einsum("i,ijk->jk", weights, np.asarray(self.focks))


def _json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _fock(core: np.ndarray, eri: np.ndarray, density: np.ndarray) -> np.ndarray:
    coulomb = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    exchange = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    return core + coulomb - 0.5 * exchange


def _diagonalize(fock: np.ndarray, orthogonalizer: np.ndarray, nocc: int):
    energies, transformed = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ transformed
    density = 2.0 * coefficients[:, :nocc] @ coefficients[:, :nocc].T
    return energies, coefficients, density


def run(spec: dict, emit: Callable[[str, dict], None], workdir: Path) -> dict:
    """Run all-electron, closed-shell RHF with explicit NumPy SCF operations."""
    started = time.perf_counter()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if spec.get("domain", "molecule") != "molecule":
        raise ValueError("교육용 RHF는 유한 분자만 지원합니다.")
    if spec.get("spin", 0) != 0 or not spec.get("restricted", True):
        raise ValueError("교육용 RHF는 spin=0인 제한 닫힌껍질 계산만 지원합니다.")
    if spec.get("method", "HF") != "HF":
        raise ValueError("교육용 계산기의 method는 HF여야 합니다. DFT를 RHF로 대체하지 않습니다.")
    if spec.get("ecp"):
        raise ValueError("교육용 RHF는 all-electron 기저만 지원하며 ECP는 지원하지 않습니다.")
    lib.num_threads(int(spec.get("threads", 2)))
    mol = gto.Mole()
    mol.atom = [(a["symbol"], a["position"]) for a in spec["atoms"]]
    mol.unit = "Angstrom"
    mol.basis = spec.get("basis", "sto-3g")
    mol.charge = int(spec.get("charge", 0))
    mol.spin = 0
    mol.verbose = 0
    mol.max_memory = int(spec.get("memory_mb", 2000))
    mol.build()
    if mol.nelectron <= 0 or mol.nelectron % 2:
        raise ValueError("RHF에는 양수인 짝수 전자수가 필요합니다.")
    nao = mol.nao_nr()
    if nao > MAX_AOS:
        raise ValueError(f"교육용 풀이기는 최대 {MAX_AOS}개 AO를 지원합니다. 현재 AO: {nao}.")
    nocc = mol.nelectron // 2
    if nocc > nao:
        raise ValueError("기저 함수 수가 점유 오비탈 수보다 작습니다.")
    # The full four-index tensor is intentionally explicit and bounded.
    emit("stage", {"name": "integrals", "message": "PySCF AO 적분 생성: S, T, V, (μν|λσ)"})
    overlap = mol.intor("int1e_ovlp")
    core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    overlap_values, overlap_vectors = np.linalg.eigh(overlap)
    if overlap_values.min() < 1e-8:
        raise ValueError("기저의 중첩 행렬이 선형 종속에 가깝습니다. 더 작은 기저를 선택하세요.")
    x = (overlap_vectors / np.sqrt(overlap_values)) @ overlap_vectors.T
    orbital_energies, coefficients, density = _diagonalize(core, x, nocc)
    nuclear_energy = float(mol.energy_nuc())
    fock = _fock(core, eri, density)
    energy = float(0.5 * np.einsum("pq,pq->", density, core + fock) + nuclear_energy)
    tolerance = float(spec.get("conv_tol", 1e-9))
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("conv_tol은 0과 1 사이의 유한한 값이어야 합니다.")
    residual_tolerance = max(1e-9, math.sqrt(tolerance) * 0.1)
    max_cycle = int(spec.get("max_cycle", 80))
    if not 1 <= max_cycle <= 500:
        raise ValueError("교육용 SCF 반복 횟수는 1~500 범위여야 합니다.")
    converged = False
    diis = _DIIS()
    history: list[dict] = []
    emit("stage", {"name": "educational_scf", "message": "NumPy RHF: 직교화 → Fock → 대각화 → 밀도 갱신"})
    with (workdir / "education_trace.jsonl").open("w", encoding="utf-8") as trace:
        for iteration in range(1, max_cycle + 1):
            residual = x.T @ (fock @ density @ overlap - overlap @ density @ fock) @ x
            extrapolated_fock = diis.extrapolate(fock, residual)
            orbital_energies, coefficients, new_density = _diagonalize(extrapolated_fock, x, nocc)
            new_fock = _fock(core, eri, new_density)
            new_energy = float(
                0.5 * np.einsum("pq,pq->", new_density, core + new_fock) + nuclear_energy
            )
            new_residual = x.T @ (
                new_fock @ new_density @ overlap - overlap @ new_density @ new_fock
            ) @ x
            change = float(np.linalg.norm(new_density - density))
            gradient = float(np.linalg.norm(new_residual))
            delta = new_energy - energy
            if not all(math.isfinite(v) for v in (new_energy, change, gradient, delta)):
                raise ValueError("SCF 과정에서 유한하지 않은 수치가 발생했습니다.")
            density, fock, energy = new_density, new_fock, new_energy
            event = {
                "iteration": iteration,
                "energy_hartree": energy,
                "delta_energy": delta,
                "gradient_norm": gradient,
                "density_change": change,
            }
            history.append(event)
            emit("scf_iteration", event)
            trace.write(json.dumps({
                **event,
                "fock": fock.tolist(),
                "density": density.tolist(),
                "coefficients": coefficients.tolist(),
                "diagonalized_fock": extrapolated_fock.tolist(),
            }, allow_nan=False) + "\n")
            trace.flush()
            if abs(delta) < tolerance and change < residual_tolerance and gradient < residual_tolerance:
                # Final canonicalization uses the actual Fock, never the DIIS
                # extrapolate. Recompute the energy from that saved density.
                eps_final, coeff_final, dm_final = _diagonalize(fock, x, nocc)
                fock_final = _fock(core, eri, dm_final)
                energy_final = float(
                    0.5 * np.einsum("pq,pq->", dm_final, core + fock_final) + nuclear_energy
                )
                residual_final = x.T @ (
                    fock_final @ dm_final @ overlap - overlap @ dm_final @ fock_final
                ) @ x
                if (abs(energy_final - energy) < tolerance
                        and np.linalg.norm(dm_final - density) < residual_tolerance
                        and np.linalg.norm(residual_final) < residual_tolerance):
                    orbital_energies, coefficients = eps_final, coeff_final
                    density, fock, energy = dm_final, fock_final, energy_final
                    converged = True
                    break

    occupations = np.zeros(nao)
    occupations[:nocc] = 2.0
    atoms = [
        {"symbol": mol.atom_symbol(i), "position": pos.tolist()}
        for i, pos in enumerate(mol.atom_coords(unit="Angstrom"))
    ]
    # Checkpoint is self-contained; volume jobs need no rerun of SCF.
    checkpoint = workdir / "educational.chk"
    lib.chkfile.save_mol(mol, str(checkpoint))
    lib.chkfile.save(str(checkpoint), "scf", {
        "e_tot": energy, "mo_energy": orbital_energies,
        "mo_coeff": coefficients, "mo_occ": occupations,
    })
    np.savez_compressed(
        workdir / "educational_arrays.npz", overlap=overlap, core=core,
        fock=fock, density=density, coefficients=coefficients,
        orbital_energies=orbital_energies, occupations=occupations,
        converged=np.array(converged),
    )
    _json_write(workdir / "education_matrices.json", {
        "overlap": overlap.tolist(), "core_hamiltonian": core.tolist(),
        "orthogonalizer": x.tolist(), "fock": fock.tolist(),
        "density": density.tolist(), "coefficients": coefficients.tolist(),
        "ao_labels": mol.ao_labels(), "basis_definition": mol._basis,
        "nuclear_repulsion_hartree": nuclear_energy,
    })
    electronic_dipole = np.einsum("xij,ji->x", mol.intor("int1e_r", comp=3), density)
    nuclear_dipole = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())
    dipole = (nuclear_dipole - electronic_dipole) * 2.541746473
    warnings = [
        "교육용 제한 HF입니다. 늘어난 결합과 강한 전자 상관을 신뢰성 있게 기술하지 못할 수 있습니다.",
        "AO 적분은 PySCF가 제공하며, 직교화·Fock 행렬·DIIS·SCF 반복은 이 프로그램에서 구현했습니다.",
    ]
    if not converged:
        warnings.append("SCF가 수렴하지 않았습니다. 중간 행렬과 에너지는 연구 결과로 사용하지 마세요.")
        emit("warning", {"message": warnings[-1]})
    return {
        "converged": converged,
        "energy_hartree": energy,
        "atoms": atoms,
        "lattice": None,
        "engine": {"name": "Quantum Lab NumPy RHF / PySCF integrals", "version": "0.1.0"},
        "orbital_energies_hartree": orbital_energies.tolist(),
        "occupations": occupations.tolist(),
        "dipole_debye": dipole.tolist(),
        "warnings": warnings,
        "iterations": len(history),
        "wall_seconds": time.perf_counter() - started,
        "provenance": {
            "pyscf": pyscf.__version__, "numpy": np.__version__,
            "method": "all-electron restricted Hartree-Fock",
            "basis": spec.get("basis", "sto-3g"),
            "basis_definition_artifact": "education_matrices.json",
            "integral_provider": "PySCF/libcint",
            "scf_implementation": "quantum_lab.engines.educational (NumPy)",
        },
        "education": {
            "integral_provider": "PySCF/libcint",
            "scf_implementation": "NumPy: symmetric orthogonalization, RHF Coulomb/exchange, DIIS",
            "ao_count": nao,
            "electron_count": float(np.einsum("ij,ji->", density, overlap)),
            "orthogonality_error": float(np.linalg.norm(coefficients.T @ overlap @ coefficients - np.eye(nao))),
            "overlap_eigenvalues": overlap_values.tolist(),
            "matrix_artifact": "education_matrices.json",
            "trace_artifact": "education_trace.jsonl",
            "equations": [
                "X = S^(-1/2); F' = X^T F X", "F = H_core + J[D] - K[D]/2",
                "D = 2 C_occ C_occ^T", "E = Tr[D(H_core + F)]/2 + E_nuc",
                "SCF iteration index is not physical time.",
            ],
        },
    }


def generate_volume(parent_dir: Path, request: dict, output_dir: Path) -> dict:
    """Generate an all-electron density or real MO CUBE from saved RHF arrays."""
    parent_dir, output_dir = Path(parent_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kind = request.get("kind", "density")
    if kind not in {"density", "orbital"}:
        raise ValueError("교육용 RHF는 전자밀도와 분자 오비탈 CUBE만 지원합니다.")
    grid = int(request.get("grid", 40))
    if not 16 <= grid <= 100:
        raise ValueError("CUBE grid는 16~100이어야 합니다.")
    if int(request.get("spin", 0)) != 0:
        raise ValueError("교육용 RHF는 공통 공간 오비탈을 사용합니다. spin 채널은 0을 선택하세요.")
    margin = 6.0
    mol = lib.chkfile.load_mol(str(parent_dir / "educational.chk"))
    with np.load(parent_dir / "educational_arrays.npz", allow_pickle=False) as arrays:
        if not bool(arrays["converged"]):
            raise ValueError("수렴하지 않은 RHF 결과에서 최종 CUBE를 생성할 수 없습니다.")
        if kind == "density":
            filename = "density.cube"
            values = cubegen.density(
                mol, str(output_dir / filename), arrays["density"], nx=grid, ny=grid, nz=grid,
                margin=margin,
            )
            units = "electron/bohr^3"
        else:
            index = int(request.get("index", 0))
            if index < 0 or index >= arrays["coefficients"].shape[1]:
                raise ValueError("오비탈 index가 범위를 벗어났습니다.")
            filename = f"orbital-{index}.cube"
            values = cubegen.orbital(
                mol, str(output_dir / filename), arrays["coefficients"][:, index],
                nx=grid, ny=grid, nz=grid, margin=margin,
            )
            units = "bohr^(-3/2)"
    return {
        "converged": True, "artifact": filename, "kind": kind, "grid": [grid] * 3,
        "units": units, "minimum": float(values.min()), "maximum": float(values.max()),
        "atoms": [
            {"symbol": mol.atom_symbol(i), "position": pos.tolist()}
            for i, pos in enumerate(mol.atom_coords(unit="Angstrom"))
        ],
        "lattice": None,
        "volume": {"kind": kind, "index": int(request.get("index", 0)), "spin": 0,
                   "grid": grid, "margin_bohr": margin, "units": units},
        "engine": {"name": "PySCF cubegen / NumPy RHF density", "version": pyscf.__version__},
        "warnings": ["유한한 표시 격자입니다. 오비탈의 두 색은 위상이며 전하 부호가 아닙니다."],
    }
