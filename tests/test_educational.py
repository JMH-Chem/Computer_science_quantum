"""Numerical reference checks for the actual educational SCF implementation."""

import json

import numpy as np
import pytest
from pyscf import gto, lib, scf

from quantum_lab.engines.educational import generate_volume, run

H2 = [{"symbol": "H", "position": [0, 0, 0]}, {"symbol": "H", "position": [0, 0, 0.74]}]
WATER = [
    {"symbol": "O", "position": [0, 0, 0]},
    {"symbol": "H", "position": [0, -0.757, 0.587]},
    {"symbol": "H", "position": [0, 0.757, 0.587]},
]


def spec(atoms, **changes):
    return {
        "domain": "molecule", "task": "educational_rhf", "atoms": atoms,
        "method": "HF", "basis": "sto-3g", "charge": 0, "spin": 0,
        "restricted": True, "threads": 1, "conv_tol": 1e-12, "max_cycle": 100,
        **changes,
    }


@pytest.mark.science
@pytest.mark.parametrize("atoms", [H2, WATER], ids=["hydrogen", "water"])
def test_numpy_rhf_matches_pyscf_energy_density_and_normalization(tmp_path, atoms):
    events = []
    result = run(spec(atoms), lambda kind, data: events.append((kind, data)), tmp_path)
    lib.num_threads(1)
    mol = gto.M(atom=[(a["symbol"], a["position"]) for a in atoms], basis="sto-3g", verbose=0)
    reference = scf.RHF(mol)
    reference.conv_tol = 1e-13
    reference.kernel()
    assert reference.converged and result["converged"]
    assert abs(result["energy_hartree"] - reference.e_tot) < 1e-8
    with np.load(tmp_path / "educational_arrays.npz") as arrays:
        assert np.max(np.abs(arrays["density"] - reference.make_rdm1())) < 1e-7
        coeff, overlap = arrays["coefficients"], arrays["overlap"]
        np.testing.assert_allclose(coeff.T @ overlap @ coeff, np.eye(mol.nao_nr()), atol=1e-10)
        assert abs(np.einsum("ij,ji->", arrays["density"], overlap) - mol.nelectron) < 1e-10
        recomputed = 0.5 * np.einsum("ij,ij->", arrays["density"], arrays["core"] + arrays["fock"])
        assert abs(recomputed + mol.energy_nuc() - result["energy_hartree"]) < 1e-12
    iterations = [data for kind, data in events if kind == "scf_iteration"]
    assert len(iterations) == result["iterations"] > 0
    trace = [json.loads(line) for line in (tmp_path / "education_trace.jsonl").read_text().splitlines()]
    assert len(trace) == len(iterations)
    assert all("density" in line and "diagonalized_fock" in line for line in trace)
    assert result["education"]["integral_provider"] == "PySCF/libcint"


@pytest.mark.science
def test_failed_convergence_retains_trace_without_final_volume(tmp_path):
    result = run(spec(WATER, max_cycle=1), lambda *_: None, tmp_path)
    assert result["converged"] is False
    assert result["iterations"] == 1
    assert (tmp_path / "education_trace.jsonl").is_file()
    with pytest.raises(ValueError, match="수렴"):
        generate_volume(tmp_path, {"kind": "density", "grid": 20}, tmp_path / "volume")


@pytest.mark.parametrize("changes", [{"spin": 2}, {"restricted": False}, {"ecp": "lanl2dz"}, {"method": "DFT"}])
def test_rejects_out_of_scope_model(tmp_path, changes):
    with pytest.raises(ValueError):
        run(spec(H2, **changes), lambda *_: None, tmp_path)


@pytest.mark.science
def test_cube_export_uses_saved_density_and_orbitals(tmp_path):
    run(spec(H2), lambda *_: None, tmp_path)
    for kind in ["density", "orbital"]:
        target = tmp_path / kind
        result = generate_volume(tmp_path, {"kind": kind, "index": 0, "grid": 24}, target)
        assert result["converged"]
        cube = target / result["artifact"]
        lines = cube.read_text().splitlines()
        assert abs(int(lines[2].split()[0])) == 2
        assert [int(lines[i].split()[0]) for i in (3, 4, 5)] == [24, 24, 24]
        values = np.fromstring(" ".join(lines[8:]), sep=" ")
        assert values.size == 24**3
        assert np.isfinite(values).all()
        assert np.max(np.abs(values)) > 0
