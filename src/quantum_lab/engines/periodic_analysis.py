"""Analysis of finite, explicitly sampled periodic Kohn--Sham eigenvalues.

Only uniform SCF mesh points enter the DOS. A high-symmetry path is useful for
band edges but does not supply Brillouin-zone integration weights.
"""
from __future__ import annotations

import numpy as np
from ase.data import atomic_masses, atomic_numbers

HARTREE_TO_EV = 27.211386245988
AMU_PER_ANGSTROM3_TO_G_PER_CM3 = 1.66053906660


def sampled_dos(energies, occupations, reference_energy_hartree: float, *, width_ev=0.1) -> dict:
    """Spin-summed Gaussian DOS per cell with equal full-mesh k weights.

    ``width_ev`` is the Gaussian standard deviation, not its FWHM. Occupied DOS
    uses the actual occupations before broadening; integrating the total DOS
    below VBM would incorrectly discard occupied Gaussian tails.
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
    relative = (energies - reference_energy_hartree) * HARTREE_TO_EV
    lower, upper = float(relative.min() - 8 * width_ev), float(relative.max() + 8 * width_ev)
    # Resolve each Gaussian with at least eight intervals per sigma. Work in
    # blocks so the number of states does not multiply peak temporary memory.
    count = max(401, int(np.ceil((upper - lower) / (width_ev / 8))) + 1)
    if count > 100_001:
        raise ValueError("DOS energy span is too large for the supported resolved display grid.")
    axis = np.linspace(lower, upper, count)
    total, occupied = np.zeros(count), np.zeros(count)
    flat_energy, flat_occ = relative.ravel(), occupations.ravel()
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
        "expected_states": 2 * norbitals,
        "electron_integral": float(np.trapezoid(occupied, axis)),
        "expected_electrons": float(occupations.sum() / npoints),
        "energy_step_ev": float(axis[1] - axis[0]),
        "units": "states/(eV cell)", "sample": "uniform SCF k mesh",
        "limitations": [
            "유한한 기저와 SCF k점 격자의 KS 상태에 Gaussian 폭을 적용한 DOS입니다. 연속적인 bulk DOS의 수렴을 보장하지 않습니다.",
            "고대칭 경로 점은 Brillouin zone 적분 가중치가 없어 DOS에 포함하지 않습니다.",
            "폭은 Gaussian 표준편차입니다. 밴드갭으로 번진 꼬리는 실제 갭 내부 상태나 온도에 따른 점유를 뜻하지 않습니다.",
            "점유 DOS는 계산된 점유수로 가중한 뒤 넓힌 값입니다. 가전자 전자 수는 이 곡선의 전체 에너지 적분과 비교합니다.",
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
        samples.append(energy)
        points.extend(kpoints.tolist())
        sources.extend([source] * len(energy))
        indices.extend(range(len(energy)))
    energy = np.concatenate(samples)
    valence, conduction = energy[:, noccupied - 1], energy[:, noccupied]
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
