"""Real periodic references: fixed input/configuration, not literature energies."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pyscf import lib
from pyscf.pbc import dft, gto

from quantum_lab.engines.periodic import HARTREE_TO_EV, _insulator_edges, generate_volume, run


def diamond_spec(kmesh=(1, 1, 1), task="bands"):
    a = 3.5668
    return {
        "name": "Diamond periodic numerical reference",
        "domain": "crystal", "task": task, "method": "DFT", "functional": "PBE",
        "atoms": [{"symbol": "C", "position": [0, 0, 0]},
                  {"symbol": "C", "position": [a / 4] * 3}],
        "lattice": [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
        "basis": "gth-szv", "pseudo": "gth-pbe", "mesh": [15, 15, 15],
        "kmesh": list(kmesh), "threads": 1, "memory_mb": 1000,
        "max_cycle": 80, "conv_tol": 1e-9, "band_points": 16,
    }


@pytest.mark.science
@pytest.mark.periodic
@pytest.mark.parametrize("kmesh", [(1, 1, 1), (2, 2, 2)])
def test_diamond_matches_direct_engine_and_density(tmp_path, kmesh):
    spec = diamond_spec(kmesh)
    events = []
    result = run(spec, lambda kind, data: events.append((kind, data)), tmp_path)
    assert result["converged"]
    assert any(kind == "scf_iteration" for kind, _ in events)
    # Deliberately assemble the reference directly with public PySCF APIs.
    lib.num_threads(1)
    cell = gto.Cell()
    cell.atom = [(a["symbol"], a["position"]) for a in spec["atoms"]]
    cell.a = spec["lattice"]
    cell.unit = "Angstrom"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.mesh = [15, 15, 15]
    cell.precision = 1e-9
    cell.verbose = 0
    cell.build()
    kpoints = cell.make_kpts(kmesh)
    reference = dft.RKS(cell) if len(kpoints) == 1 else dft.KRKS(cell, kpts=kpoints)
    reference.xc = "PBE"
    reference.conv_tol = 1e-9
    reference.max_memory = 1000
    reference.grids.mesh = cell.mesh
    energy = reference.kernel()
    assert reference.converged
    assert abs(result["energy_hartree"] - energy) / cell.natm < 1e-6

    band = result["bands"]
    points = cell.get_abs_kpts(np.asarray(band["kpoints_scaled"]))
    reference_bands = np.asarray(reference.get_bands(points)[0])
    nocc = cell.nelectron // 2
    reference_vbm = max(np.atleast_2d(reference.mo_energy)[:, :nocc].max(),
                        reference_bands[:, :nocc].max())
    np.testing.assert_allclose(band["energies_ev"],
                               (reference_bands - reference_vbm) * HARTREE_TO_EV,
                               atol=1e-4, rtol=0)
    assert band["gap_ev"] > 0
    assert band["reference"] == "VBM"
    assert "not an optical gap" in band["gap_kind"]
    assert len(band["distance"]) == len(band["energies_ev"])
    expected_breaks = [
        i for i in range(1, len(band["distance"]))
        if abs(band["distance"][i] - band["distance"][i - 1]) < 1e-10
        and np.linalg.norm(np.asarray(band["kpoints_scaled"])[i]
                           - np.asarray(band["kpoints_scaled"])[i - 1]) > 1e-10
    ]
    assert expected_breaks  # Diamond's ASE path contains disconnected segments.
    assert band["break_indices"] == expected_breaks
    assert any("|" in label["label"] for label in band["labels"])
    assert all(b["distance"] - a["distance"] > 1e-10
               for a, b in zip(band["labels"], band["labels"][1:]))

    (tmp_path / "input.json").write_text(json.dumps(spec))
    (tmp_path / "result.json").write_text(json.dumps(result))
    volume = generate_volume(tmp_path, {"kind": "density", "grid": 24}, tmp_path / "volume")
    assert (tmp_path / "volume" / volume["artifact"]).is_file()
    # Integrate valence density over one skewed unit cell, including all k weights.
    assert abs(volume["electron_integral"] - cell.nelectron) < 0.03
    assert volume["expected_valence_electrons"] == 8


@pytest.mark.science
@pytest.mark.periodic
def test_unconverged_periodic_run_does_not_claim_band_success(tmp_path):
    spec = diamond_spec()
    spec["max_cycle"] = 1
    events = []
    result = run(spec, lambda kind, data: events.append((kind, data)), tmp_path)
    assert result["converged"] is False
    assert "bands" not in result
    assert any(kind == "warning" for kind, _ in events)
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="수렴한"):
        generate_volume(tmp_path, {"kind": "density"}, tmp_path / "volume")


def test_periodic_rejects_unsupported_models_before_computing(tmp_path):
    for update in ({"functional": "B3LYP"}, {"charge": 1}, {"task": "tddft"},
                   {"basis": "def2-svp"}, {"lattice": [[0, 0, 0]] * 3}):
        spec = diamond_spec() | update
        with pytest.raises(ValueError):
            run(spec, lambda *_: None, tmp_path)
    with pytest.raises(ValueError, match="가전자 밀도"):
        generate_volume(tmp_path, {"kind": "orbital"}, tmp_path)


def test_metallic_and_fractional_occupations_are_not_called_insulators():
    with pytest.raises(ValueError, match="부분 점유"):
        _insulator_edges(np.array([[-1.0, 0.1], [-0.5, 0.3]]),
                         np.array([[2.0, 0.0], [1.5, 0.5]]), 1)
    with pytest.raises(ValueError, match="비금속"):
        _insulator_edges(np.array([[-0.1, 0.1], [-0.5, -0.2]]),
                         np.array([[2.0, 0.0], [2.0, 0.0]]), 1)
