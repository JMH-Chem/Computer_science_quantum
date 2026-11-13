"""State-count sum rules and actual periodic engine checks for DOS and gaps."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from pyscf import lib
from pyscf.pbc import dft, gto

from quantum_lab.engines.periodic import run
from quantum_lab.engines.periodic_analysis import (
    HARTREE_TO_EV,
    crystal_properties,
    sampled_band_edges,
    sampled_dos,
)


def diamond_spec(kmesh=(1, 1, 1), task="bands"):
    a = 3.5668
    return {
        "domain": "crystal", "task": task, "method": "DFT", "functional": "PBE",
        "atoms": [{"symbol": "C", "position": [0, 0, 0]},
                  {"symbol": "C", "position": [a / 4] * 3}],
        "lattice": [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
        "basis": "gth-szv", "pseudo": "gth-pbe", "mesh": [15] * 3,
        "kmesh": list(kmesh), "threads": 1, "memory_mb": 1000,
        "max_cycle": 80, "conv_tol": 1e-9, "band_points": 16,
    }


def test_dos_spin_k_weights_and_occupied_sum_rules():
    energies = np.array([[-0.4, 0.1], [-0.3, 0.2]])
    occupations = np.array([[2.0, 0], [2.0, 0]])
    result = sampled_dos(energies, occupations, -0.3)
    axis = np.array(result["energy_ev"])
    total = np.array(result["density_states_per_ev"])
    occupied = np.array(result["occupied_density_states_per_ev"])
    assert result["state_integral"] == pytest.approx(4, abs=1e-10)
    assert result["electron_integral"] == pytest.approx(2, abs=1e-10)
    assert result["expected_states"] == 4 and result["expected_electrons"] == 2
    # The first moment is unchanged by a normalized symmetric broadening.
    assert np.trapezoid(axis * occupied, axis) == pytest.approx(-0.1 * HARTREE_TO_EV, abs=1e-9)
    assert np.all(total >= 0) and np.all(occupied <= total + 1e-12)
    assert result["kpoint_weights"] == [0.5, 0.5]
    assert result["reference"] == "SCF-mesh VBM"
    # Duplicating the full mesh must not double either count or intensity.
    twice = sampled_dos(np.tile(energies, (2, 1)), np.tile(occupations, (2, 1)), -0.3)
    np.testing.assert_allclose(twice["density_states_per_ev"], total, atol=1e-12)
    np.testing.assert_allclose(twice["occupied_density_states_per_ev"], occupied, atol=1e-12)


def test_dos_gamma_width_is_standard_deviation_and_not_fermi_smearing():
    result = sampled_dos([0.0, 1.0], [2, 0], 0.0)
    axis = np.array(result["energy_ev"])
    occupied = np.array(result["occupied_density_states_per_ev"])
    variance = np.trapezoid(axis**2 * occupied, axis) / result["electron_integral"]
    assert variance == pytest.approx(0.1**2, abs=1e-10)
    assert result["spin_degeneracy"] == 2
    assert result["kpoint_weights"] == [1]
    assert "표준편차" in " ".join(result["limitations"])


@pytest.mark.parametrize("energies,occupations,reference,width", [
    ([[float("nan"), 1]], [[2, 0]], 0, 0.1),
    ([[-1, 1]], [[2]], 0, 0.1),
    ([[-1, 1]], [[3, 0]], 0, 0.1),
    ([[-1, 1]], [[2, 0]], float("inf"), 0.1),
    ([[-1, 1]], [[2, 0]], 0, 0),
])
def test_invalid_dos_inputs_rejected(energies, occupations, reference, width):
    with pytest.raises(ValueError):
        sampled_dos(energies, occupations, reference, width_ev=width)


def test_gap_extrema_include_path_without_treating_it_as_a_uniform_mesh():
    result = sampled_band_edges([[-0.5, 0.2], [-0.4, 0.5]],
                                [[0, 0, 0], [0.5, 0, 0]], 1,
                                path_energies=[[-0.3, 0.1], [-0.2, 0.3]],
                                path_points=[[0.25, 0, 0], [0.1, 0, 0]])
    assert result["indirect_gap_ev"] == pytest.approx(0.3 * HARTREE_TO_EV)
    assert result["direct_gap_ev"] == pytest.approx(0.4 * HARTREE_TO_EV)
    assert result["vbm"]["source"] == result["cbm"]["source"] == "band_path"
    assert result["vbm"]["index"] == 1 and result["cbm"]["index"] == 0
    assert result["vbm"]["band_index"] == 0 and result["cbm"]["band_index"] == 1
    assert result["direct_gap_location"]["kpoint_scaled"] == [0.25, 0, 0]
    assert result["classification"] == "indirect on sampled points"
    assert "not an optical" in result["gap_kind"]
    gamma = sampled_band_edges([[-0.5, 0.2]], [[0, 0, 0]], 1)
    assert gamma["classification"] == "direct on sampled points"
    assert gamma["sample"] == "SCF mesh"
    # Γ alone is direct only in that sample, not proof of a direct-gap crystal.
    assert "Brillouin" in gamma["limitations"]


def test_gap_rejects_unordered_and_mismatched_arrays():
    with pytest.raises(ValueError, match="ascending"):
        sampled_band_edges([[0.1, -0.2]], [[0, 0, 0]], 1)
    with pytest.raises(ValueError, match="corresponding"):
        sampled_band_edges([[-0.1, 0.2]], [[0, 0]], 1)


def test_crystal_properties_use_full_nuclear_mass_and_primitive_volume():
    spec = diamond_spec()
    result = crystal_properties(spec, -10)
    assert result["atom_count"] == 2
    assert result["energy_per_atom_hartree"] == -5
    assert result["volume_angstrom3"] == pytest.approx(3.5668**3 / 4)
    assert 3.50 < result["density_g_cm3"] < 3.53
    inverted = {**spec, "lattice": (-np.array(spec["lattice"])).tolist()}
    assert crystal_properties(inverted, -10)["density_g_cm3"] == result["density_g_cm3"]


@pytest.mark.science
@pytest.mark.periodic
@pytest.mark.parametrize("kmesh", [(1, 1, 1), (2, 2, 2)])
def test_actual_diamond_dos_and_gap_match_independent_engine(tmp_path, kmesh):
    spec = diamond_spec(kmesh)
    events = []
    result = run(spec, lambda kind, data: events.append((kind, data)), tmp_path)
    assert result["converged"] and result["scf_converged"]
    assert result["method"] == "DFT" and result["task"] == "bands"
    assert result["iteration_count"] == sum(kind == "scf_iteration" for kind, _ in events)
    assert result["basis_functions"] == result["n_basis"]
    for name in ("basis_provider", "basis_provider_version", "libxc_version", "numpy_version"):
        assert result["provenance"][name]
    lib.num_threads(1)
    cell = gto.Cell()
    cell.atom = [(a["symbol"], a["position"]) for a in spec["atoms"]]
    cell.a, cell.unit = spec["lattice"], "Angstrom"
    cell.basis, cell.pseudo = "gth-szv", "gth-pbe"
    cell.mesh, cell.precision, cell.verbose = [15] * 3, 1e-9, 0
    cell.build()
    points = cell.make_kpts(kmesh)
    reference = dft.RKS(cell) if len(points) == 1 else dft.KRKS(cell, kpts=points)
    reference.xc, reference.conv_tol = "PBE", 1e-9
    reference.max_memory, reference.grids.mesh = 1000, cell.mesh
    reference.kernel()
    assert reference.converged
    energies = np.atleast_2d(reference.mo_energy)
    nocc = cell.nelectron // 2
    vbm = float(energies[:, :nocc].max())
    dos = result["sampled_dos"]
    assert dos["kpoint_count"] == int(np.prod(kmesh))
    assert dos["state_integral"] == pytest.approx(2 * cell.nao_nr(), abs=1e-8)
    assert dos["electron_integral"] == pytest.approx(cell.nelectron, abs=1e-8)
    assert dos["reference_energy_hartree"] == pytest.approx(vbm, abs=1e-8)
    # Evaluate the analytic broadened sum at sparse points using independent
    # engine eigenvalues. Neither the helper nor saved adapter eigenvalues enter.
    sigma = dos["width_ev"]
    for index in np.linspace(0, len(dos["energy_ev"]) - 1, 57, dtype=int):
        x = dos["energy_ev"][index]
        expected = sum(2 * np.exp(-((x - (e - vbm) * HARTREE_TO_EV) / sigma)**2 / 2)
                       / (len(points) * sigma * np.sqrt(2 * np.pi)) for e in energies.ravel())
        assert dos["density_states_per_ev"][index] == pytest.approx(expected, abs=1e-5)
    path = np.array(result["bands"]["kpoints_scaled"])
    path_energy = np.asarray(reference.get_bands(cell.get_abs_kpts(path))[0])
    combined = np.concatenate([energies, path_energy])
    edge = result["sampled_band_edges"]
    assert edge["indirect_gap_ev"] == pytest.approx(
        (combined[:, nocc:].min() - combined[:, :nocc].max()) * HARTREE_TO_EV, abs=1e-4,
    )
    assert edge["direct_gap_ev"] == pytest.approx(
        (combined[:, nocc] - combined[:, nocc - 1]).min() * HARTREE_TO_EV, abs=1e-4,
    )
    assert edge["indirect_gap_ev"] == pytest.approx(result["bands"]["gap_ev"], abs=1e-10)
    with (tmp_path / "sampled-dos.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == len(dos["energy_ev"])
    assert float(exported[20]["dos_states_per_ev_per_cell"]) == dos["density_states_per_ev"][20]
    assert json.loads((tmp_path / "band-edges.json").read_text()) == edge
    assert result["crystal_properties"]["energy_per_atom_hartree"] == result["energy_hartree"] / 2


@pytest.mark.science
@pytest.mark.periodic
def test_unconverged_periodic_job_keeps_analysis_unavailable(tmp_path):
    result = run({**diamond_spec(), "max_cycle": 1}, lambda *_: None, tmp_path)
    assert result["converged"] is False
    assert result["scf_converged"] is False
    assert "sampled_dos" not in result and "sampled_band_edges" not in result
    assert not (tmp_path / "sampled-dos.csv").exists()
