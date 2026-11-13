"""Small insulating crystals with Gaussian-orbital periodic PBE.

The FFT grid is an auxiliary representation: this adapter is not a plane-wave
orbital solver.  All band gaps reported here are sampled Kohn--Sham gaps, not
optical excitation energies or proof of an insulating bulk material.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyscf
from ase import Atoms
from pyscf import lib
from pyscf.dft import libxc
from pyscf.pbc import dft, gto
from pyscf.pbc.dft import numint
from pyscf.pbc.lib import chkfile
from pyscf.tools import cubegen

from .periodic_analysis import crystal_properties, sampled_band_edges, sampled_dos

HARTREE_TO_EV = 27.211386245988


def _number(value):
    if value is None:
        return None
    number = float(np.real(value))
    return number if np.isfinite(number) else None


def _validate(spec: dict) -> None:
    if spec.get("method", "DFT") != "DFT":
        raise ValueError("주기계 첫 버전은 PBE DFT만 지원합니다.")
    if spec.get("functional", "PBE").upper() != "PBE":
        raise ValueError("주기계 첫 버전에서 검증하는 함수형은 PBE입니다.")
    if spec.get("charge", 0) or spec.get("spin", 0) or not spec.get("restricted", True):
        raise ValueError("주기계는 중성·닫힌껍질 3차원 결정만 지원합니다.")
    if spec.get("ecp"):
        raise ValueError("주기계에서는 분자 ECP 대신 GTH-PBE 가상퍼텐셜을 선택하세요.")
    if not isinstance(spec.get("basis"), str) or spec["basis"].lower() not in {"gth-szv", "gth-dzvp"}:
        raise ValueError("주기계 기저는 GTH-SZV 또는 GTH-DZVP를 선택하세요.")
    if spec.get("pseudo", "gth-pbe").lower() != "gth-pbe":
        raise ValueError("PBE와 함께 검증한 가상퍼텐셜은 GTH-PBE입니다.")
    if spec.get("task", "single_point") not in {"single_point", "bands"}:
        raise ValueError("주기계는 고정 구조 단일점 계산과 밴드 구조를 지원합니다.")
    lattice = np.asarray(spec.get("lattice"), dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) < 1e-6:
        raise ValueError("부피가 0이 아닌 3차원 격자 벡터가 필요합니다.")
    for key, minimum in (("kmesh", 1), ("mesh", 5)):
        values = np.asarray(spec.get(key, [1, 1, 1] if key == "kmesh" else [15, 15, 15]))
        if values.shape != (3,) or (values < minimum).any() or not np.equal(values, values.astype(int)).all():
            raise ValueError(f"{key}는 {minimum} 이상의 정수 3개여야 합니다.")


def _cell(spec: dict) -> gto.Cell:
    cell = gto.Cell()
    cell.atom = [(a["symbol"], a["position"]) for a in spec["atoms"]]
    cell.a = spec["lattice"]
    cell.unit = "Angstrom"
    cell.basis = spec["basis"]
    cell.pseudo = spec.get("pseudo", "gth-pbe")
    cell.dimension = 3
    cell.charge = 0
    cell.spin = 0
    cell.mesh = spec.get("mesh", [15, 15, 15])
    cell.precision = min(1e-8, float(spec.get("conv_tol", 1e-9)))
    cell.max_memory = int(spec.get("memory_mb", 2000))
    cell.verbose = 3
    cell.build()
    if cell.nelectron % 2:
        raise ValueError("닫힌껍질 계산에는 짝수의 가전자 수가 필요합니다.")
    if cell.nao_nr() <= cell.nelectron // 2:
        raise ValueError("선택한 기저에 비점유 상태가 없습니다. 더 큰 기저를 선택하세요.")
    return cell


def _insulator_edges(energies: np.ndarray, occupations: np.ndarray, nocc: int) -> tuple[float, float]:
    energies = np.atleast_2d(energies)
    occupations = np.atleast_2d(occupations)
    expected = np.zeros_like(occupations)
    expected[:, :nocc] = 2
    if not np.allclose(occupations, expected, atol=1e-8, rtol=0):
        raise ValueError(
            "k점마다 점유 밴드 수가 다르거나 부분 점유가 있습니다. "
            "금속·스미어링 계산은 이 버전에서 지원하지 않습니다."
        )
    vbm = float(energies[:, :nocc].max())
    cbm = float(energies[:, nocc:].min())
    if cbm - vbm <= 1e-7:
        raise ValueError(
            "표본 k점에서 비금속 밴드갭을 확인하지 못했습니다. "
            "금속 또는 밴드 중첩을 지원하는 엔진이 필요합니다."
        )
    return vbm, cbm


def run(spec: dict, emit: Callable[[str, dict], None], workdir: Path) -> dict:
    """Run real PBE SCF and, optionally, a sampled ASE high-symmetry band path."""
    _validate(spec)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    lib.num_threads(int(spec.get("threads", 2)))
    emit("stage", {"name": "build", "message": "주기적 Gaussian 기저와 GTH-PBE 가상퍼텐셜 준비"})
    cell = _cell(spec)
    kmesh = np.asarray(spec.get("kmesh", [1, 1, 1]), dtype=int)
    kpts = cell.make_kpts(kmesh)
    gamma = len(kpts) == 1
    mf = dft.RKS(cell) if gamma else dft.KRKS(cell, kpts=kpts)
    mf.xc = "PBE"
    mf.max_cycle = int(spec.get("max_cycle", 80))
    mf.conv_tol = float(spec.get("conv_tol", 1e-9))
    mf.max_memory = int(spec.get("memory_mb", 2000))
    mf.chkfile = str(workdir / "checkpoint.chk")
    mf.grids.mesh = cell.mesh
    iteration_count = 0

    def callback(env):
        nonlocal iteration_count
        iteration_count += 1
        e = _number(env.get("e_tot"))
        last = _number(env.get("last_hf_e"))
        emit("scf_iteration", {
            "iteration": int(env["cycle"]) + 1,
            "energy_hartree": e,
            "delta_energy": e - last if e is not None and last is not None else None,
            "gradient_norm": _number(env.get("norm_gorb")),
            "density_change": _number(env.get("norm_ddm")),
        })

    mf.callback = callback
    emit("stage", {"name": "scf", "message": f"PBE SCF: {len(kpts)}개의 k점"})
    energy = mf.kernel()
    warnings = [
        "작은 k점·격자·기저는 교육용 시작값입니다. 물성 해석 전에 수렴성을 확인하세요.",
        "Gaussian 궤도 기저를 사용합니다. FFT 격자는 평면파 궤도 기저가 아닙니다.",
    ]
    result = {
        "converged": bool(mf.converged),
        "scf_converged": bool(mf.converged),
        "method": "DFT", "task": spec.get("task", "single_point"),
        "energy_hartree": _number(energy),
        "atoms": spec["atoms"],
        "lattice": spec["lattice"],
        "engine": {"name": "PySCF PBC", "version": pyscf.__version__},
        "electron_count": cell.nelectron,
        "n_basis": cell.nao_nr(),
        "basis_functions": cell.nao_nr(), "iteration_count": iteration_count,
        "kmesh": kmesh.tolist(),
        "kpoints_scaled": cell.get_scaled_kpts(kpts).tolist(),
        "mesh": np.asarray(cell.mesh).tolist(),
        "orbital_energies_hartree": np.asarray(mf.mo_energy).tolist(),
        "occupations": np.asarray(mf.mo_occ).tolist(),
        "warnings": warnings,
        "units": {"energy": "hartree", "coordinates": "angstrom", "bands": "eV"},
        "provenance": {
            "orbital_basis": spec["basis"], "pseudopotential": "gth-pbe", "functional": "PBE",
            "density_fitting": "FFTDF", "boundary": "3D periodic",
            "basis_source": "https://github.com/pyscf/pyscf/tree/master/pyscf/pbc/gto/basis",
            "pseudo_source": "https://github.com/pyscf/pyscf/tree/master/pyscf/pbc/gto/pseudo",
            "conv_tol": mf.conv_tol,
            "basis_provider": "PySCF periodic built-in", "basis_provider_version": pyscf.__version__,
            "libxc_version": libxc.__version__, "numpy_version": np.__version__,
        },
    }
    (workdir / "periodic_basis.json").write_text(
        json.dumps({"basis": cell._basis, "pseudo": cell._pseudo}, indent=2), encoding="utf-8"
    )
    dm = np.asarray(mf.make_rdm1())
    np.savez_compressed(workdir / "periodic_state.npz", density=dm, kpoints=kpts,
                        mo_coeff=np.asarray(mf.mo_coeff), mo_occ=np.asarray(mf.mo_occ))
    # Save the final post-check state, which may differ from the final callback.
    mf.dump_chk(str(workdir / "checkpoint.chk"))
    if not mf.converged:
        emit("warning", {"message": "SCF가 수렴하지 않아 밴드·밀도 해석을 진행하지 않습니다."})
        result["elapsed_seconds"] = time.monotonic() - started
        return result

    nocc = cell.nelectron // 2
    vbm, cbm = _insulator_edges(np.asarray(mf.mo_energy), np.asarray(mf.mo_occ), nocc)
    result["sampled_scf_gap_ev"] = (cbm - vbm) * HARTREE_TO_EV
    emit("stage", {"name": "periodic_analysis", "message": "SCF k점 DOS·표본 밴드 경계·단위격자 물성 계산"})
    result["crystal_properties"] = crystal_properties(spec, energy)
    result["sampled_dos"] = sampled_dos(mf.mo_energy, mf.mo_occ, vbm)
    result["sampled_band_edges"] = sampled_band_edges(
        mf.mo_energy, cell.get_scaled_kpts(kpts), nocc,
    )
    warnings.append("DOS는 균등 SCF k점에 Gaussian 폭을 적용한 표시입니다. 밴드 경로는 DOS 적분에 사용하지 않습니다.")
    if spec.get("task") == "bands":
        emit("stage", {"name": "bands", "message": "ASE 고대칭 경로의 Kohn–Sham 밴드 계산"})
        atoms = Atoms(symbols=[a["symbol"] for a in spec["atoms"]],
                      positions=[a["position"] for a in spec["atoms"]], cell=spec["lattice"], pbc=True)
        path = atoms.cell.bandpath(npoints=int(spec.get("band_points", 40)))
        abs_kpoints = cell.get_abs_kpts(path.kpts)
        band_energy, _ = mf.get_bands(abs_kpoints)
        band_energy = np.asarray(band_energy)
        distance, label_distance, labels = path.get_linear_kpoint_axis()
        # ASE assigns equal abscissae to the end and beginning of disconnected
        # path segments. Keep physical samples untouched and expose where plots
        # must break; otherwise the viewer draws fictitious vertical bands.
        break_indices = [
            i for i in range(1, len(distance))
            if np.isclose(distance[i], distance[i - 1], rtol=0, atol=1e-10)
            and not np.allclose(path.kpts[i], path.kpts[i - 1], rtol=0, atol=1e-10)
        ]
        axis_labels = []
        for x, label in zip(label_distance, labels):
            label = "Γ" if label == "G" else label
            if axis_labels and np.isclose(axis_labels[-1]["distance"], x, rtol=0, atol=1e-10):
                if label not in axis_labels[-1]["label"].split("|"):
                    axis_labels[-1]["label"] += "|" + label
            else:
                axis_labels.append({"distance": float(x), "label": label})
        # Include SCF mesh and path extrema; a path gap is still only sampled.
        vbm = max(vbm, float(band_energy[:, :nocc].max()))
        cbm = min(cbm, float(band_energy[:, nocc:].min()))
        gap = (cbm - vbm) * HARTREE_TO_EV
        if gap <= 1e-7:
            raise ValueError("밴드 경로에서 밴드 중첩이 관찰되었습니다. 첫 버전의 비금속 지원 범위를 벗어납니다.")
        result["bands"] = {
            "distance": distance.tolist(),
            "energies_ev": ((band_energy - vbm) * HARTREE_TO_EV).tolist(),
            "labels": axis_labels,
            "break_indices": break_indices,
            "reference": "VBM",
            "reference_energy_hartree": vbm,
            "gap_ev": gap,
            "gap_kind": "sampled Kohn–Sham gap (SCF mesh and band path); not an optical gap",
            "n_occupied": nocc,
            "path": path.path,
            "kpoints_scaled": path.kpts.tolist(),
            "distance_units": "angstrom^-1",
        }
        result["sampled_band_edges"] = sampled_band_edges(
            mf.mo_energy, cell.get_scaled_kpts(kpts), nocc,
            path_energies=band_energy, path_points=path.kpts,
        )
    with (workdir / "sampled-dos.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["energy_ev_relative_to_scf_vbm", "dos_states_per_ev_per_cell",
                         "occupied_dos_electrons_per_ev_per_cell"])
        dos = result["sampled_dos"]
        writer.writerows(zip(dos["energy_ev"], dos["density_states_per_ev"],
                            dos["occupied_density_states_per_ev"], strict=True))
    (workdir / "band-edges.json").write_text(
        json.dumps(result["sampled_band_edges"], ensure_ascii=False, indent=2), encoding="utf-8",
    )
    result["analysis_artifacts"] = ["sampled-dos.csv", "band-edges.json"]
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def generate_volume(parent_dir: Path, request: dict, output_dir: Path) -> dict:
    """Evaluate the k-point-averaged valence density over one periodic cell.

    Bloch orbitals can be complex; first-version periodic exports deliberately
    expose total valence density only, not misleading molecular orbital plots.
    """
    parent_dir, output_dir = Path(parent_dir), Path(output_dir)
    if request.get("kind", "density") != "density":
        raise ValueError("주기계는 가전자 밀도만 CUBE로 제공합니다. 분자 오비탈·NTO는 분자 계산에서 사용하세요.")
    result_path = parent_dir / "result.json"
    if result_path.exists() and not json.loads(result_path.read_text())["converged"]:
        raise ValueError("수렴한 주기계 계산에서만 밀도를 생성할 수 있습니다.")
    spec_path = parent_dir / "input.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    lib.num_threads(int(spec.get("threads", 2)))
    grid = int(request.get("grid", 40))
    if not 8 <= grid <= 120:
        raise ValueError("밀도 격자는 축마다 8~120점 범위여야 합니다.")
    cell = chkfile.load_cell(str(parent_dir / "checkpoint.chk"))
    with np.load(parent_dir / "periodic_state.npz") as state:
        dm, kpoints = state["density"], state["kpoints"]
    if dm.ndim == 2:
        dm = dm[np.newaxis, :, :]
    if len(dm) != len(kpoints):
        raise ValueError("저장된 밀도 행렬과 k점 수가 일치하지 않습니다.")
    cube = cubegen.Cube(cell, nx=grid, ny=grid, nz=grid)
    coords = cube.get_coords()
    rho = np.zeros(len(coords))
    for start in range(0, len(coords), 1024):
        end = min(len(coords), start + 1024)
        aos = cell.pbc_eval_gto("GTOval_sph", coords[start:end], kpts=kpoints)
        for ao, density in zip(aos, dm):
            rho[start:end] += numint.eval_rho(cell, ao, density, hermi=1).real / len(kpoints)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = f"density-{grid}.cube"
    cube.write(rho.reshape(grid, grid, grid), str(output_dir / artifact),
               comment="Periodic k-averaged valence electron density (e/Bohr^3); not all-electron core density")
    integral = float(rho.mean() * cell.vol)
    return {
        "converged": True, "artifact": artifact, "kind": "density", "grid": [grid] * 3,
        "electron_integral": integral, "expected_valence_electrons": cell.nelectron,
        "integration_error": integral - cell.nelectron,
        "units": "electron/bohr^3", "domain": "crystal",
        "warnings": ["가상퍼텐셜의 가전자 밀도입니다. 핵심부 전자를 포함하지 않습니다.",
                     "적분값은 표시 격자 오차를 포함합니다. 격자를 조밀하게 하여 수렴성을 확인하세요."],
        "engine": {"name": "PySCF PBC", "version": pyscf.__version__},
    }
