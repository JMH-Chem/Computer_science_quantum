"""Compare app adapter outputs with independent direct PySCF calls.

Basis definitions are reconstructed from the recorded BSE artifact. PySCF's
built-in basis aliases may have rounded coefficients, so they are not an exact
same-input reference for the BSE-resolved adapter.
"""

import json

import numpy as np
import pytest
from pyscf import dft, gto, lib, scf
from scipy.constants import physical_constants

from quantum_lab.engines.molecular import generate_volume, run

EV_PER_HARTREE = physical_constants["Hartree energy in eV"][0]
H2 = [{"symbol": "H", "position": [0, 0, 0]}, {"symbol": "H", "position": [0, 0, 0.74]}]
WATER = [
    {"symbol": "O", "position": [0, 0, 0]},
    {"symbol": "H", "position": [0, -0.757, 0.587]},
    {"symbol": "H", "position": [0, 0.757, 0.587]},
]
FORMALDEHYDE = [
    {"symbol": "C", "position": [0, 0, 0]},
    {"symbol": "O", "position": [0, 0, 1.208]},
    {"symbol": "H", "position": [0, 0.943, -0.588]},
    {"symbol": "H", "position": [0, -0.943, -0.588]},
]


def spec(atoms=H2, **changes):
    return {
        "domain": "molecule", "task": "single_point", "atoms": atoms,
        "method": "HF", "basis": "sto-3g", "functional": "PBE", "charge": 0, "spin": 0,
        "restricted": True, "threads": 1, "conv_tol": 1e-10, "max_cycle": 80,
        "grid_level": 1, "nstates": 3, "max_steps": 30, **changes,
    }


def direct_reference(settings, directory, atoms=None):
    """Construct and execute direct PySCF independently of adapter helpers."""
    lib.num_threads(1)
    saved = json.loads((directory / "basis.json").read_text())
    mol = gto.M(
        atom=[(a["symbol"], a["position"]) for a in (atoms or settings["atoms"])],
        unit="Angstrom", basis=saved["resolved_pyscf_basis"], ecp=saved["resolved_pyscf_ecp"],
        charge=settings["charge"], spin=settings["spin"], verbose=0,
    )
    if settings["method"] == "DFT":
        ref = dft.RKS(mol) if settings["restricted"] else dft.UKS(mol)
        ref.xc = settings["functional"]
        ref.grids.level = settings["grid_level"]
    else:
        ref = scf.RHF(mol) if settings["restricted"] else scf.UHF(mol)
    ref.conv_tol = settings["conv_tol"]
    ref.max_cycle = settings["max_cycle"]
    ref.kernel()
    return ref


@pytest.mark.science
@pytest.mark.parametrize("settings", [spec(), spec(WATER, method="DFT")], ids=["H2-HF", "water-PBE"])
def test_ground_state_matches_direct_engine_and_preserves_basis(settings, tmp_path):
    events = []
    result = run(settings, lambda kind, data: events.append((kind, data)), tmp_path)
    reference = direct_reference(settings, tmp_path)
    assert result["converged"] and reference.converged
    assert abs(result["energy_hartree"] - reference.e_tot) < 1e-8
    np.testing.assert_allclose(result["orbital_energies_hartree"], reference.mo_energy, atol=1e-8)
    assert abs(result["integrated_electrons_ao"] - reference.mol.nelectron) < 1e-8
    saved = json.loads((tmp_path / "basis.json").read_text())
    assert saved["provider"] == "Basis Set Exchange"
    assert set(saved["elements"]) == {a["symbol"] for a in settings["atoms"]}
    assert (tmp_path / "basis-references.txt").stat().st_size > 0
    assert any(kind == "scf_iteration" for kind, _ in events)
    assert result["provenance"]["time_dynamics"] is False


@pytest.mark.science
@pytest.mark.parametrize("method", ["HF", "DFT"])
def test_unrestricted_doublet_matches_direct_engine(method, tmp_path):
    settings = spec([{"symbol": "H", "position": [0, 0, 0]}], method=method, spin=1, restricted=False)
    result = run(settings, lambda *_: None, tmp_path)
    reference = direct_reference(settings, tmp_path)
    assert result["converged"] and reference.converged
    assert abs(result["energy_hartree"] - reference.e_tot) < 1e-8
    assert np.asarray(result["occupations"]).shape == (2, 1)
    assert np.sum(result["occupations"]) == 1


@pytest.mark.science
@pytest.mark.parametrize("task,conv_tol", [("tda", 1e-10), ("tda", 1e-12), ("tddft", 1e-10)],
                         ids=["tda", "tda-tight", "tddft"])
def test_excited_states_match_direct_roots_strengths_and_normalized_ntos(task, conv_tol, tmp_path):
    settings = spec(FORMALDEHYDE, method="DFT", task=task, conv_tol=conv_tol)
    result = run(settings, lambda *_: None, tmp_path)
    reference = direct_reference(settings, tmp_path)
    td = reference.TDA() if task == "tda" else reference.TDDFT()
    td.nstates = settings["nstates"]
    td.singlet = True
    td.conv_tol = settings["conv_tol"]
    td.max_cycle = settings["max_cycle"]
    td.kernel()
    assert result["converged"] and np.all(td.converged)
    assert len(result["excitations"]) == settings["nstates"]
    np.testing.assert_allclose(
        [x["energy_ev"] for x in result["excitations"]], np.asarray(td.e) * EV_PER_HARTREE,
        atol=1e-4, rtol=0,
    )
    np.testing.assert_allclose(
        [x["oscillator_strength"] for x in result["excitations"]],
        td.oscillator_strength(gauge="length"), atol=1e-7, rtol=0,
    )
    if task == "tda":
        # Independently diagonalize the complete small response matrix. Tiny
        # Davidson dependence cutoffs can produce spurious roots even when
        # the ground-state reference and requested tolerance are unchanged.
        response_matrix, _ = reference.TDA().get_ab()
        dimension = np.prod(response_matrix.shape[:2])
        dense_roots = np.linalg.eigvalsh(response_matrix.reshape(dimension, dimension))
        np.testing.assert_allclose(
            [state["energy_ev"] for state in result["excitations"]],
            dense_roots[:settings["nstates"]] * EV_PER_HARTREE, atol=1e-4, rtol=0,
        )
        vind, _ = reference.TDA().gen_vind()
        saved = lib.chkfile.load(str(tmp_path / "checkpoint.chk"), "tddft")
        for energy, pair in zip(saved["e"], saved["xy"], strict=True):
            vector = np.asarray(pair[0]).ravel()
            vector = vector / np.linalg.norm(vector)
            residual = np.asarray(vind(vector[None, :]))[0] - float(energy) * vector
            assert np.linalg.norm(residual) < settings["conv_tol"]
    overlap = reference.mol.intor("int1e_ovlp")
    with np.load(tmp_path / "orbitals.npz") as arrays:
        for index, state in enumerate(result["excitations"]):
            assert state["energy_ev"] > 0
            assert state["converged"]
            assert abs(sum(state["nto_weights"]) - 1) < 1e-10
            for kind in ["nto_hole", "nto_particle"]:
                coeff = arrays[f"{kind}_{index}"]
                np.testing.assert_allclose(coeff.T @ overlap @ coeff, np.eye(coeff.shape[1]), atol=1e-8)
    assert "X-only" in result["provenance"]["nto_convention"]
    volume = generate_volume(tmp_path, {"kind": "nto_hole", "index": 0, "grid": 20}, tmp_path / "nto")
    assert volume["volume"]["weight"] == result["excitations"][0]["nto_weights"][0]


@pytest.mark.science
def test_optimization_records_real_geometries_and_final_energy(tmp_path):
    atoms = [{"symbol": "H", "position": [0, 0, 0]}, {"symbol": "H", "position": [0, 0, 1.1]}]
    settings = spec(atoms, task="optimize")
    events = []
    result = run(settings, lambda kind, data: events.append((kind, data)), tmp_path)
    assert result["converged"] and result["geometry_converged"]
    assert len(result["optimization"]) > 1
    assert result["energy_hartree"] < result["optimization"][0]["energy_hartree"]
    positions = np.array([a["position"] for a in result["atoms"]])
    assert 0.65 < np.linalg.norm(positions[1] - positions[0]) < 0.85
    direct = direct_reference(settings, tmp_path, atoms=result["atoms"])
    assert abs(result["energy_hartree"] - direct.e_tot) < 1e-8
    steps = [data for kind, data in events if kind == "optimization_step"]
    assert len(steps) == len(result["optimization"])
    assert all(item["scf_converged"] for item in steps)


@pytest.mark.science
def test_unconverged_scf_is_not_a_success_or_excited_state_result(tmp_path):
    events = []
    result = run(spec(WATER, method="DFT", task="tda", max_cycle=1),
                 lambda kind, data: events.append((kind, data)), tmp_path)
    assert result["converged"] is False
    assert result["scf_converged"] is False
    assert result["excitations"] == []
    assert any(kind == "warning" for kind, _ in events)
    assert any(kind == "scf_iteration" for kind, _ in events)
    with pytest.raises(ValueError, match="converged"):
        generate_volume(tmp_path, {"kind": "density", "grid": 20}, tmp_path / "volume")


def _cube_integral(path, square=False):
    lines = path.read_text().splitlines()
    natoms = abs(int(lines[2].split()[0]))
    dimensions = [int(lines[i].split()[0]) for i in (3, 4, 5)]
    steps = np.array([[float(x) for x in lines[i].split()[1:]] for i in (3, 4, 5)])
    values = np.fromstring(" ".join(lines[6 + natoms:]), sep=" ").reshape(dimensions)
    assert np.isfinite(values).all()
    if square:
        values = values**2
    for _ in range(3):
        values = np.trapezoid(values, axis=0)
    return float(values * abs(np.linalg.det(steps)))


@pytest.mark.science
def test_volume_density_integrates_and_mo_is_normalized_with_finite_box_tolerance(tmp_path):
    result = run(spec(), lambda *_: None, tmp_path)
    assert result["converged"]
    density = generate_volume(tmp_path, {"kind": "density", "grid": 48}, tmp_path / "density")
    orbital = generate_volume(tmp_path, {"kind": "orbital", "index": 0, "grid": 48}, tmp_path / "orbital")
    electrons = _cube_integral(tmp_path / "density" / density["artifact"])
    norm = _cube_integral(tmp_path / "orbital" / orbital["artifact"], square=True)
    # Finite 6-bohr-margin, 48^3 grid: explicit 0.25% quadrature/tail allowance.
    # This tests the exported values and actual voxel units, not only metadata.
    assert density["volume"]["margin_bohr"] == 6
    assert abs(electrons - 2) < 0.005
    assert abs(norm - 1) < 0.0025
    assert abs(electrons - density["volume"]["grid_integrated_electrons"]) < 2e-5


@pytest.mark.science
def test_element_specific_basis_is_preserved(tmp_path):
    settings = spec(WATER, basis={"H": "def2-svp", "O": "sto-3g"})
    result = run(settings, lambda *_: None, tmp_path)
    direct = direct_reference(settings, tmp_path)
    assert result["converged"]
    assert abs(result["energy_hartree"] - direct.e_tot) < 1e-8
    saved = json.loads((tmp_path / "basis.json").read_text())
    assert saved["elements"]["H"]["basis"]["name"].lower() == "def2-svp"
    assert saved["elements"]["O"]["basis"]["name"].lower() == "sto-3g"


@pytest.mark.science
def test_default_formaldehyde_four_roots_meet_requested_response_residual(tmp_path):
    """Regression: tiny Davidson corrections must not be discarded too early."""
    from quantum_lab.examples import examples

    settings = next(example["spec"] for example in examples() if example["id"] == "formaldehyde")
    settings["threads"] = 1
    assert settings["nstates"] == 4
    result = run(settings, lambda *_: None, tmp_path)
    assert result["converged"] and result["excited_converged"]
    assert len(result["excitations"]) == 4
    assert all(state["converged"] for state in result["excitations"])
    assert result["provenance"]["response_solver"]["conv_tol"] == settings["conv_tol"]
    assert result["provenance"]["response_solver"]["max_cycle"] == settings["max_cycle"]
    # Evaluate the saved eigenvectors against a separately built response
    # operator. This checks actual residuals instead of trusting a boolean flag.
    reference = direct_reference(settings, tmp_path)
    response = reference.TDA()
    vind, _ = response.gen_vind()
    saved = lib.chkfile.load(str(tmp_path / "checkpoint.chk"), "tddft")
    assert len(saved["xy"]) == 4
    for energy, pair in zip(saved["e"], saved["xy"], strict=True):
        vector = np.asarray(pair[0]).ravel()
        vector = vector / np.linalg.norm(vector)
        residual = np.asarray(vind(vector[None, :]))[0] - float(energy) * vector
        assert np.linalg.norm(residual) < settings["conv_tol"]


@pytest.mark.science
@pytest.mark.parametrize("task,method,functional", [
    ("tda", "HF", "PBE"), ("tdhf", "HF", "PBE"),
    ("tda", "DFT", "PBE"), ("tddft", "DFT", "PBE"), ("tddft", "DFT", "PBE0"),
])
def test_lowest_roots_cover_sectors_missing_from_koopmans_guesses(task, method, functional, tmp_path):
    """Two formaldehyde HF-TDA guesses omit the lower B1 state entirely."""
    settings = spec(FORMALDEHYDE, method=method, functional=functional, task=task, nstates=2)
    result = run(settings, lambda *_: None, tmp_path)
    assert result["converged"] and result["excited_converged"]
    search = result["provenance"]["response_solver"]["root_search"]
    assert search["complete"] and search["strategy"] == "abelian_symmetry_blocks"
    assert {block["irrep"] for block in search["blocks"]} == {"A1", "A2", "B1", "B2"}
    assert all(block["complete"] for block in search["blocks"])
    reference = direct_reference(settings, tmp_path)
    response = reference.TDA() if task == "tda" else reference.TDHF() if task == "tdhf" else reference.TDDFT()
    a, b = response.get_ab()
    dimension = np.prod(a.shape[:2])
    a, b = a.reshape(dimension, dimension), b.reshape(dimension, dimension)
    if task == "tda":
        complete = np.linalg.eigvalsh(a)
    else:
        roots = np.linalg.eigvals(np.block([[a, b], [-b, -a]]))
        complete = np.sort(roots[(abs(roots.imag) < 1e-8) & (roots.real > 1e-4)].real)
    np.testing.assert_allclose([s["energy_ev"] for s in result["excitations"]],
                               complete[:2] * EV_PER_HARTREE, atol=1e-4, rtol=0)
    assert all(s["residual_norm"] < settings["conv_tol"] for s in result["excitations"])
    if task == "tda" and method == "HF":
        response.nstates, response.conv_tol, response.lindep = 2, 1e-10, np.finfo(float).eps
        response.kernel()
        assert np.all(response.converged)
        # Low residuals alone accepted a root almost 3 eV above the missing one.
        assert (response.e[1] - complete[1]) * EV_PER_HARTREE > 2
        assert [s["symmetry"] for s in result["excitations"]] == ["A2", "B1"]


@pytest.mark.science
def test_unconverged_unselected_sector_prevents_lowest_state_success(tmp_path, monkeypatch):
    from pyscf.tdscf import rhf
    original = rhf.TDA.kernel

    def incomplete_sector(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if self.wfnsym == 0:  # A1 is not the requested lowest A2 state.
            self.converged = np.zeros_like(self.converged, dtype=bool)
        return value

    monkeypatch.setattr(rhf.TDA, "kernel", incomplete_sector)
    result = run(spec(FORMALDEHYDE, task="tda", nstates=1), lambda *_: None, tmp_path)
    assert result["excitations"][0]["converged"]
    assert not result["converged"] and not result["excited_converged"]
    assert not result["provenance"]["response_solver"]["root_search"]["complete"]


@pytest.mark.science
def test_unlabelled_small_response_uses_complete_coordinate_fallback(tmp_path, monkeypatch):
    from quantum_lab.engines import response

    def mixed(*_args, **_kwargs):
        raise ValueError("Mixed orbitals in this regression fixture")

    monkeypatch.setattr(response, "orbital_symmetry_labels", mixed)
    result = run(spec(FORMALDEHYDE, task="tda", nstates=2), lambda *_: None, tmp_path)
    assert result["converged"]
    search = result["provenance"]["response_solver"]["root_search"]
    assert search["complete"] and search["strategy"] == "full_coordinate_fallback"
    reference = direct_reference(spec(FORMALDEHYDE), tmp_path)
    matrix, _ = reference.TDA().get_ab()
    matrix = matrix.reshape(32, 32)
    np.testing.assert_allclose([s["energy_ev"] for s in result["excitations"]],
                               np.linalg.eigvalsh(matrix)[:2] * EV_PER_HARTREE, atol=1e-4, rtol=0)


def test_large_unlabelled_response_is_unavailable_without_partial_success(monkeypatch):
    from types import SimpleNamespace

    from quantum_lab.engines import response

    def mixed(*_args, **_kwargs):
        raise ValueError("Actual orbitals mix symmetry sectors")

    monkeypatch.setattr(response, "orbital_symmetry_labels", mixed)
    mf = SimpleNamespace(mol=None, mo_coeff=np.eye(50), mo_occ=np.array([2] * 8 + [0] * 42),
                         mo_energy=np.arange(50, dtype=float))
    td = SimpleNamespace(_scf=mf, conv_tol=1e-10, chkfile=None)
    search = response.solve_response(td, 2, lambda *_: None)
    assert search["strategy"] == "unavailable" and not search["complete"]
    assert not search["blocks"] and not td.xy and td.e.size == 0


@pytest.mark.science
@pytest.mark.parametrize("kind", ["tda", "casida"])
def test_small_positive_response_roots_are_not_removed_by_engine_filter(kind):
    """A complete trial basis must not silently discard positive low roots."""
    from quantum_lab.engines.response import solve_response
    from quantum_lab.engines.symmetry import orbital_symmetry_labels

    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    reference = dft.RKS(molecule) if kind == "casida" else scf.RHF(molecule)
    if kind == "casida":
        reference.xc = "PBE"
        reference.grids.level = 1
    reference.conv_tol = 1e-10
    reference.kernel()
    td = reference.TDDFT() if kind == "casida" else reference.TDA()
    td.conv_tol, td.lindep, td.chkfile = 1e-9, np.finfo(float).eps, None
    labels, _ = orbital_symmetry_labels(molecule, reference.mo_coeff)
    pairs = (labels[reference.mo_occ == 2, None] ^ labels[reference.mo_occ == 0]).ravel()
    same_sector = np.flatnonzero(pairs == pairs[0])
    assert len(same_sector) == 2
    energies = np.full(len(pairs), 0.3)
    energies[same_sector] = [1e-4, 0.2]
    diagonal = energies**2 if kind == "casida" else energies
    # An exactly known positive diagonal operator exposes filtering, separate
    # from the chemical reference or any iterative convergence uncertainty.
    td.gen_vind = lambda *args: (lambda vectors: np.asarray(vectors) * diagonal, diagonal.copy())
    search = solve_response(td, 1, lambda *_: None)
    assert search["complete"] and td.converged.all()
    np.testing.assert_allclose(td.e, [1e-4], rtol=0, atol=1e-12)
    assert search["selected_states"][0]["residual_norm"] < td.conv_tol
    cutoff = search["positive_root_filter"]
    assert cutoff["operator_eigenvalue_threshold"] == np.finfo(float).eps
    assert cutoff["operator_eigenvalue_unit"] == ("hartree_squared" if kind == "casida" else "hartree")
    assert cutoff["excitation_energy_threshold_hartree"] < td.e[0]
