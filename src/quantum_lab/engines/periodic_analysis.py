"""Analysis of finite, explicitly sampled periodic Kohn--Sham eigenvalues.

Only uniform SCF mesh points enter the DOS. A high-symmetry path is useful for
band edges but does not supply Brillouin-zone integration weights.
"""
from __future__ import annotations

import numpy as np
from ase.data import atomic_masses, atomic_numbers

HARTREE_TO_EV = 27.211386245988
AMU_PER_ANGSTROM3_TO_G_PER_CM3 = 1.66053906660


class DOSResolutionError(ValueError):
    """Physical eigenvalues exceed the resolved DOS display grid budget."""


def retained_orbital_mask(energies, coefficients, invalid_energy: float) -> np.ndarray:
    """Identify engine padding, never infer missing orbitals from a large energy.

    PySCF pads rank-reduced periodic eigensystems with its exact invalid-energy
    constant AND all-zero coefficient columns. Neither condition alone proves
    that a state is padding. The returned mask always has a k-point dimension.
    """
    energies = np.atleast_2d(np.asarray(energies, dtype=float))
    coefficients = np.asarray(coefficients)
    if coefficients.ndim == 2:
        coefficients = coefficients[None, ...]
    if (energies.ndim != 2 or not energies.size or coefficients.ndim != 3
            or coefficients.shape[0] != energies.shape[0]
            or coefficients.shape[2] != energies.shape[1] or not coefficients.shape[1]
            or not np.isfinite(energies).all() or not np.isfinite(coefficients).all()
            or not np.isfinite(invalid_energy)):
        raise ValueError("Orbital energies and coefficients must be finite matching eigensystems.")
    sentinel = energies == invalid_energy
    zero_column = np.all(coefficients == 0, axis=1)
    if not np.array_equal(sentinel, zero_column):
        raise ValueError("Invalid orbital padding requires both the engine sentinel and zero coefficients.")
    valid = ~sentinel
    if (not valid.any(axis=1).all()
            or (np.diff(valid.astype(int), axis=1) > 0).any()):
        raise ValueError("Retained orbitals must precede trailing engine padding at every k point.")
    return valid


def sampled_dos(energies, occupations, reference_energy_hartree: float, *, width_ev=0.1,
                valid_orbitals=None) -> dict:
    """Spin-summed Gaussian DOS per cell with equal full-mesh k weights.

    ``width_ev`` is the Gaussian standard deviation, not its FWHM. Occupied DOS
    uses the actual occupations before broadening; integrating the total DOS
    below VBM would incorrectly discard occupied Gaussian tails. A supplied
    mask excludes proven engine padding, retaining the original k-point weights.
    """
    energies = np.atleast_2d(np.asarray(energies, dtype=float))
    occupations = np.atleast_2d(np.asarray(occupations, dtype=float))
    if (energies.ndim != 2 or occupations.shape != energies.shape or not energies.size
            or not np.isfinite(energies).all() or not np.isfinite(occupations).all()):
        raise ValueError("DOS requires finite, matching k-point energy and occupation arrays.")
    if (occupations < 0).any() or (occupations > 2).any():
        raise ValueError("Restricted spin-summed occupations must be between zero and two.")
    if not np.isfinite(reference_energy_hartree) or not np.isfinite(width_ev) or width_ev <= 0:
        raise ValueError("DOS reference and positive Gaussian width must be finite.")
    npoints, norbitals = energies.shape
    valid = np.ones(energies.shape, dtype=bool) if valid_orbitals is None else np.asarray(valid_orbitals)
    if (valid.shape != energies.shape or valid.dtype != np.dtype(bool)
            or not valid.any(axis=1).all()):
        raise ValueError("DOS requires a boolean retained-orbital mask for every k point.")
    if np.any(occupations[~valid] != 0):
        raise ValueError("Occupied states cannot be excluded from the DOS as engine padding.")
    relative = (energies[valid] - reference_energy_hartree) * HARTREE_TO_EV
    lower, upper = float(relative.min() - 8 * width_ev), float(relative.max() + 8 * width_ev)
    # Resolve each Gaussian with at least eight intervals per sigma. Work in
    # blocks so the number of states does not multiply peak temporary memory.
    count = max(401, int(np.ceil((upper - lower) / (width_ev / 8))) + 1)
    if count > 100_001:
        raise DOSResolutionError("DOS energy span is too large for the supported resolved display grid.")
    axis = np.linspace(lower, upper, count)
    total, occupied = np.zeros(count), np.zeros(count)
    flat_energy, flat_occ = relative, occupations[valid]
    prefactor = 1 / (width_ev * np.sqrt(2 * np.pi) * npoints)
    for first in range(0, len(flat_energy), 64):
        last = first + 64
        scaled = (axis[:, None] - flat_energy[None, first:last]) / width_ev
        gaussian = np.exp(-0.5 * scaled**2) * prefactor
        total += 2 * gaussian.sum(axis=1)
        occupied += gaussian @ flat_occ[first:last]
    return {
        "energy_ev": axis.tolist(),
        "density_states_per_ev": total.tolist(),
        "occupied_density_states_per_ev": occupied.tolist(),
        "width_ev": float(width_ev), "broadening": "Gaussian sigma",
        "kpoint_count": npoints, "kpoint_weights": [1 / npoints] * npoints,
        "spin_degeneracy": 2, "reference": "SCF-mesh VBM",
        "reference_energy_hartree": float(reference_energy_hartree),
        "state_integral": float(np.trapezoid(total, axis)),
        "expected_states": float(2 * valid.sum() / npoints),
        "nominal_states": 2 * norbitals,
        "retained_orbitals_per_kpoint": valid.sum(axis=1).tolist(),
        "electron_integral": float(np.trapezoid(occupied, axis)),
        "expected_electrons": float(occupations.sum() / npoints),
        "energy_step_ev": float(axis[1] - axis[0]),
        "units": "states/(eV cell)", "sample": "uniform SCF k mesh",
        "limitations": [
            "유한한 기저와 SCF k점 격자의 KS 상태에 Gaussian 폭을 적용한 DOS입니다. 연속적인 bulk DOS의 수렴을 보장하지 않습니다.",
            "고대칭 경로 점은 Brillouin zone 적분 가중치가 없어 DOS에 포함하지 않습니다.",
            "폭은 Gaussian 표준편차입니다. 밴드갭으로 번진 꼬리는 실제 갭 내부 상태나 온도에 따른 점유를 뜻하지 않습니다.",
            "점유 DOS는 계산된 점유수로 가중한 뒤 넓힌 값입니다. 가전자 전자 수는 이 곡선의 전체 에너지 적분과 비교합니다.",
            "겹침 행렬의 선형 종속성으로 엔진이 제거한 궤도 슬롯은 상태 수에 포함하지 않습니다. 적분 목표는 k점 가중치를 적용한 실제 보존 상태 수입니다.",
        ],
    }


def sampled_band_edges(scf_energies, scf_points, noccupied: int, *,
                       path_energies=None, path_points=None) -> dict:
    """Direct and indirect gaps over the supplied points, without BZ claims."""
    samples, points, sources, indices = [], [], [], []
    datasets = [("scf_mesh", scf_energies, scf_points)]
    if path_energies is not None:
        datasets.append(("band_path", path_energies, path_points))
    for source, energy, kpoints in datasets:
        energy = np.atleast_2d(np.asarray(energy, dtype=float))
        kpoints = np.asarray(kpoints, dtype=float)
        if (energy.ndim != 2 or kpoints.shape != (len(energy), 3) or not energy.size
                or not 0 < noccupied < energy.shape[1] or not np.isfinite(energy).all()
                or not np.isfinite(kpoints).all()):
            raise ValueError("Band edges require finite energies and corresponding scaled k points.")
        if (np.diff(energy, axis=1) < -1e-10).any():
            raise ValueError("Band eigenvalues must be sorted in ascending order at each k point.")
        # Different samples may retain different numbers of virtual orbitals
        # after overlap rank reduction. Only the occupied/conduction frontier
        # enters these extrema; no absent high-energy columns are needed.
        samples.append(energy[:, noccupied - 1:noccupied + 1])
        points.extend(kpoints.tolist())
        sources.extend([source] * len(energy))
        indices.extend(range(len(energy)))
    energy = np.concatenate(samples)
    valence, conduction = energy[:, 0], energy[:, 1]
    iv, ic = int(np.argmax(valence)), int(np.argmin(conduction))
    vbm, cbm = float(valence[iv]), float(conduction[ic])
    direct = (conduction - valence) * HARTREE_TO_EV
    idir = int(np.argmin(direct))
    indirect_gap = (cbm - vbm) * HARTREE_TO_EV
    direct_gap = float(direct[idir])

    def location(i):
        return {"kpoint_scaled": points[i], "source": sources[i], "index": indices[i]}

    return {
        "sample": "SCF mesh + high-symmetry path" if path_energies is not None else "SCF mesh",
        "indirect_gap_ev": indirect_gap, "direct_gap_ev": direct_gap,
        "vbm": {**location(iv), "energy_hartree": vbm, "energy_ev": 0.0,
                "band_index": noccupied - 1},
        "cbm": {**location(ic), "energy_hartree": cbm, "energy_ev": indirect_gap,
                "band_index": noccupied},
        "direct_gap_location": location(idir),
        "classification": ("direct on sampled points" if abs(direct_gap - indirect_gap) < 1e-6
                           else "indirect on sampled points"),
        "reference": "VBM",
        "gap_kind": "sampled Kohn–Sham eigenvalue gap; not an optical or quasiparticle gap",
        "index_convention": "zero-based band and source-local k-point indices",
        "limitations": "유한한 표본 k점 밖의 극값은 포함하지 않으므로 전체 Brillouin zone의 직접·간접 갭을 확정하지 않습니다.",
    }


def crystal_properties(spec: dict, energy_hartree: float) -> dict:
    """Cell geometry and chemical mass; independent of pseudopotential cores."""
    count = len(spec["atoms"])
    volume = abs(float(np.linalg.det(np.asarray(spec["lattice"], dtype=float))))
    if count == 0 or not np.isfinite(volume) or volume <= 0:
        raise ValueError("Crystal properties require atoms and a nonzero finite cell volume.")
    mass = sum(float(atomic_masses[atomic_numbers[a["symbol"]]]) for a in spec["atoms"])
    return {
        "atom_count": count, "volume_angstrom3": volume,
        "energy_per_atom_hartree": float(energy_hartree) / count,
        "density_g_cm3": mass * AMU_PER_ANGSTROM3_TO_G_PER_CM3 / volume,
        "mass_convention": "ASE standard atomic masses; no isotope-specific mass selection",
    }
