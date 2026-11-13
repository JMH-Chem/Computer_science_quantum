"""Real correlated and HF-response calculations, checked against direct PySCF."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import cc, ci, fci, gto, lib, mcscf, mp, mrpt, scf
from scipy.constants import physical_constants

from quantum_lab.engines.correlated import run_correlation, run_eom, validate_space
from quantum_lab.engines.molecular import generate_volume, run

H2 = [{"symbol": "H", "position": [0, 0, 0]}, {"symbol": "H", "position": [0, 0, 0.74]}]
WATER = [{"symbol": "O", "position": [0, 0, 0]},
         {"symbol": "H", "position": [0, -0.757, 0.587]},
         {"symbol": "H", "position": [0, 0.757, 0.587]}]


def spec(method="MP2", **changes):
    return {"domain": "molecule", "task": "single_point", "method": method, "atoms": H2,
            "basis": "sto-3g", "charge": 0, "spin": 0, "restricted": True, "threads": 1,
            "conv_tol": 1e-10, "max_cycle": 80, "memory_mb": 2000,
            "active_electrons": 2, "active_orbitals": 2, "frozen_orbitals": 0, **changes}


def reference(settings, path):
    lib.num_threads(1)
    saved = json.loads((path / "basis.json").read_text())
    mol = gto.M(atom=[(a["symbol"], a["position"]) for a in settings["atoms"]],
                unit="Angstrom", basis=saved["resolved_pyscf_basis"], ecp=saved["resolved_pyscf_ecp"],
                charge=settings["charge"], spin=settings["spin"], verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = settings["conv_tol"]
    mf.max_cycle = settings["max_cycle"]
    mf.kernel()
    assert mf.converged
    return mf


def direct_energy(mf, settings):
    method, frozen = settings["method"], settings["frozen_orbitals"] or None
    if method == "MP2":
        solver = mp.MP2(mf, frozen=frozen)
        solver.kernel()
        return solver.e_tot
    if method in {"CCSD", "CCSD(T)"}:
        solver = cc.CCSD(mf, frozen=frozen)
        solver.conv_tol = settings["conv_tol"]
        solver.max_cycle = settings["max_cycle"]
        solver.kernel()
        assert solver.converged
        return solver.e_tot + (solver.ccsd_t() if method == "CCSD(T)" else 0)
    if method == "CISD":
        solver = ci.CISD(mf, frozen=frozen)
        solver.conv_tol = settings["conv_tol"]
        solver.max_cycle = settings["max_cycle"]
        solver.kernel()
        assert solver.converged
        return solver.e_tot
    if method == "FCI":
        solver = fci.FCI(mf, singlet=True)
        solver.conv_tol = settings["conv_tol"]
        solver.max_cycle = settings["max_cycle"]
        value, _ = solver.kernel()
        assert solver.converged
        return value
    factory = mcscf.CASSCF if method == "CASSCF" else mcscf.CASCI
    solver = factory(mf, settings["active_orbitals"], settings["active_electrons"])
    solver.fcisolver = fci.direct_spin0.FCI(mf.mol)
    solver.fcisolver.conv_tol = settings["conv_tol"]
    solver.fcisolver.max_cycle = settings["max_cycle"]
    if method == "CASSCF":
        solver.conv_tol = settings["conv_tol"]
        solver.max_cycle_macro = settings["max_cycle"]
    solver.kernel()
    assert solver.converged
    return solver.e_tot + (mrpt.NEVPT(solver, root=0).kernel() if method == "NEVPT2" else 0)


@pytest.mark.science
@pytest.mark.parametrize("method", ["MP2", "CCSD", "CCSD(T)", "CISD", "FCI", "CASCI", "CASSCF", "NEVPT2"])
def test_h2_methods_match_direct_engine_and_preserve_reference_distinction(method, tmp_path):
    settings = spec(method)
    events = []
    result = run(settings, lambda kind, data: events.append((kind, data)), tmp_path)
    mf = reference(settings, tmp_path)
    assert result["converged"] and result["scf_converged"]
    assert result["correlation"]["converged"]
    expected = direct_energy(mf, settings)
    assert abs(result["energy_hartree"] - expected) < 1e-8
    assert abs(result["reference_energy_hartree"] - mf.e_tot) < 1e-10
    assert abs(result["energy_hartree"] - result["reference_energy_hartree"]
               - result["correlation_energy_hartree"]) < 1e-12
    assert result["correlation_energy_hartree"] < 0
    for field in ("orbital_reference", "density_reference", "dipole_reference"):
        assert "RHF reference" in result[field] and "not correlated" in result[field]
    np.testing.assert_allclose(result["orbital_energies_hartree"], mf.mo_energy, atol=1e-9)
    assert (tmp_path / "correlation-wavefunction.npz").is_file()
    assert any(kind == "scf_iteration" for kind, _ in events)
    assert any(kind == "stage" and data.get("name") == "correlation_complete" for kind, data in events)
    correlation = result["correlation"]
    if method in {"CISD", "FCI", "CASCI", "CASSCF", "NEVPT2"}:
        assert abs(sum(correlation["natural_occupations"]) - 2) < 1e-8
    if method in {"CASCI", "CASSCF", "NEVPT2"}:
        assert correlation["active_space"]["initial_orbital_indices_1based"] == [1, 2]
    if method == "CCSD(T)":
        assert abs(correlation["diagnostics"]["triples_energy_hartree"]) < 1e-12


@pytest.mark.science
@pytest.mark.parametrize("method", ["MP2", "CCSD", "CCSD(T)", "CISD"])
def test_frozen_core_water_matches_direct_same_electronic_definition(method, tmp_path):
    settings = spec(method, atoms=WATER, frozen_orbitals=1)
    result = run(settings, lambda *_: None, tmp_path)
    mf = reference(settings, tmp_path)
    assert result["converged"]
    assert result["correlation"]["frozen_orbitals"] == 1
    assert abs(result["energy_hartree"] - direct_energy(mf, settings)) < 1e-8
    if method == "CCSD(T)":
        assert result["correlation"]["diagnostics"]["triples_energy_hartree"] < 0


@pytest.mark.science
@pytest.mark.parametrize("method", ["FCI", "CASCI", "CASSCF", "NEVPT2"])
def test_water_multiconfigurational_spaces_match_direct_engine(method, tmp_path):
    settings = spec(method, atoms=WATER)
    result = run(settings, lambda *_: None, tmp_path)
    mf = reference(settings, tmp_path)
    assert result["converged"]
    assert abs(result["energy_hartree"] - direct_energy(mf, settings)) < 1e-8
    correlation = result["correlation"]
    assert abs(correlation["diagnostics"]["spin_square"]) < 1e-8
    expected_electrons = 10 if method == "FCI" else 2
    assert abs(sum(correlation["natural_occupations"]) - expected_electrons) < 1e-8
    if method != "FCI":
        assert correlation["active_space"]["initial_orbital_indices_1based"] == [5, 6]
        assert correlation["active_space"]["core_orbitals"] == 4
    if method == "NEVPT2":
        assert correlation["diagnostics"]["perturbation_reference"] == "CASCI"
        assert correlation["diagnostics"]["nevpt2_correction_hartree"] < -0.001
        assert abs(correlation["diagnostics"]["casci_energy_hartree"]
                   + correlation["diagnostics"]["nevpt2_correction_hartree"] - result["energy_hartree"]) < 1e-10


@pytest.mark.science
def test_correlation_volume_exports_standalone_reference_label(tmp_path):
    result = run(spec(), lambda *_: None, tmp_path)
    volume = generate_volume(tmp_path, {"kind": "density", "grid": 20}, tmp_path / "volume")
    assert result["converged"] and volume["converged"]
    assert "RHF reference" in volume["volume"]["density_kind"]
    assert "RHF reference" in volume["volume"]["reference"]
    first = (tmp_path / "volume" / volume["artifact"]).read_text().splitlines()[0]
    assert "RHF reference" in first and "not a correlated" in first


@pytest.mark.science
@pytest.mark.parametrize("method", ["CCSD", "CCSD(T)", "CISD"])
def test_nonconverged_correlated_solver_keeps_intermediate_separate(method, tmp_path):
    settings = spec(method, atoms=WATER)
    # Converge the reference first so max_cycle=1 tests the post-HF stage.
    run({**settings, "method": "HF"}, lambda *_: None, tmp_path)
    mf = reference(settings, tmp_path)
    correlation = run_correlation(mf, {**settings, "max_cycle": 1}, lambda *_: None, tmp_path)
    assert not correlation["converged"]
    assert correlation["total_energy_hartree"] is not None
    assert any("did not converge" in warning for warning in correlation["warnings"])
    if method == "CCSD(T)":
        assert correlation["diagnostics"]["triples_energy_hartree"] is None


@pytest.mark.science
def test_nonconverged_reference_never_claims_correlated_energy(tmp_path):
    result = run(spec(atoms=WATER, max_cycle=1), lambda *_: None, tmp_path)
    assert not result["converged"] and not result["scf_converged"]
    assert result["energy_hartree"] is None
    assert result["reference_energy_hartree"] is not None
    assert result["correlation"] is None
    assert not (tmp_path / "correlation-wavefunction.npz").exists()


@pytest.mark.science
@pytest.mark.parametrize("task", ["tda", "tdhf"])
def test_hf_singlet_response_matches_direct_engine_and_nto(task, tmp_path):
    settings = spec("HF", task=task, nstates=1)
    result = run(settings, lambda *_: None, tmp_path)
    mf = reference(settings, tmp_path)
    response = mf.TDA() if task == "tda" else mf.TDHF()
    response.nstates, response.singlet = 1, True
    response.conv_tol = settings["conv_tol"]
    response.kernel()
    assert result["converged"] and result["excited_converged"]
    np.testing.assert_allclose([x["energy_ev"] for x in result["excitations"]],
                               response.e * physical_constants["Hartree energy in eV"][0], atol=1e-4, rtol=0)
    np.testing.assert_allclose([x["oscillator_strength"] for x in result["excitations"]],
                               response.oscillator_strength(), atol=1e-8, rtol=0)
    assert abs(sum(result["excitations"][0]["nto_weights"]) - 1) < 1e-10
    nto = generate_volume(tmp_path, {"kind": "nto_hole", "index": 0, "grid": 20}, tmp_path / "nto")
    assert nto["converged"] and nto["volume"]["state"] == 1


@pytest.mark.science
def test_water_eomccsd_roots_and_saved_residuals_match_direct_reference(tmp_path):
    settings = spec("CCSD", atoms=WATER, task="eomccsd", nstates=3)
    result = run(settings, lambda *_: None, tmp_path)
    mf = reference(settings, tmp_path)
    correlated = cc.CCSD(mf)
    correlated.conv_tol = settings["conv_tol"]
    correlated.kernel()
    response = correlated.EOMEESinglet()
    response.conv_tol = settings["conv_tol"]
    energies, _ = response.kernel(nroots=3)
    assert result["converged"] and result["eom"]["converged"] and np.all(response.converged)
    # The total energy remains the ground-state CC energy; roots are differences.
    assert abs(result["energy_hartree"] - correlated.e_tot) < 1e-8
    assert result["excitations"] == []  # no made-up transition strengths or TDDFT NTOs
    states = result["eom"]["states"]
    np.testing.assert_allclose([state["energy_ev"] for state in states],
                               energies * physical_constants["Hartree energy in eV"][0], atol=1e-4, rtol=0)
    matvec, _ = response.gen_matvec()
    with np.load(tmp_path / "eom-wavefunction.npz", allow_pickle=False) as data:
        for vector, energy, state in zip(data["right_vectors"], data["energies_hartree"], states, strict=True):
            vector = vector / np.linalg.norm(vector)
            residual = np.linalg.norm(np.asarray(matvec([vector]))[0] - energy * vector)
            assert residual < result["eom"]["residual_tolerance"]
            assert "oscillator_strength" not in state and "nto_weights" not in state


@pytest.mark.science
def test_eom_rejects_unavailable_root_count_instead_of_clamping(tmp_path):
    with pytest.raises(ValueError, match="excitation-vector dimension"):
        run(spec("CCSD", task="eomccsd", nstates=3), lambda *_: None, tmp_path)


@pytest.mark.science
def test_eom_failure_is_distinct_from_ground_state_convergence(tmp_path):
    settings = spec("CCSD", atoms=WATER, task="eomccsd", nstates=3)
    run({**settings, "method": "HF", "task": "single_point"}, lambda *_: None, tmp_path)
    mf = reference(settings, tmp_path)
    correlated = cc.CCSD(mf).run(conv_tol=1e-10)
    assert correlated.converged
    response = run_eom(correlated, {**settings, "max_cycle": 1}, lambda *_: None, tmp_path)
    assert not response["converged"]
    assert any(not state["converged"] for state in response["states"])
    ground_failure = run_correlation(mf, {**settings, "max_cycle": 1}, lambda *_: None, tmp_path)
    assert not ground_failure["converged"] and ground_failure["eom"] is None
    assert any("EOM-EE-CCSD was not run" in warning for warning in ground_failure["warnings"])


@pytest.mark.parametrize("method,nmo,electrons,changes,message", [
    ("FCI", 13, 2, {}, "12 molecular orbitals"),
    ("FCI", 12, 12, {}, "determinants"),
    ("CASCI", 12, 12, {"active_orbitals": 12, "active_electrons": 12}, "determinants"),
    ("CASCI", 7, 10, {"active_orbitals": 4, "active_electrons": 2}, "Core plus active"),
    ("CASSCF", 7, 10, {"active_orbitals": 2, "active_electrons": 3}, "even active"),
    ("MP2", 7, 10, {"frozen_orbitals": 5}, "occupied"),
    ("FCI", 7, 10, {"frozen_orbitals": 1}, "Frozen"),
    ("MP2", 100, 10, {"memory_mb": 256}, "memory budget"),
    ("NEVPT2", 10, 2, {"active_orbitals": 9, "active_electrons": 2}, "8 active orbitals"),
])
def test_resource_and_space_guards_run_before_large_transform(method, nmo, electrons, changes, message):
    mf = SimpleNamespace(mol=SimpleNamespace(spin=0, nelectron=electrons), mo_coeff=np.empty((nmo, nmo)))
    with pytest.raises(ValueError, match=message):
        validate_space(mf, spec(method, **changes))


@pytest.mark.parametrize("changes", [{"task": "optimize"}, {"restricted": False}, {"spin": 2}])
def test_correlated_unsupported_task_or_reference_is_not_substituted(changes, tmp_path):
    with pytest.raises(ValueError, match="closed-shell RHF single-point"):
        run(spec(**changes), lambda *_: None, tmp_path)
