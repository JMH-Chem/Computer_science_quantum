"""Numerical checks for populations, projected modes, IR derivatives and RRHO."""
import csv
import json

import numpy as np
import pytest
from pyscf import dft, gto, lib, scf
from pyscf.hessian import thermo
from scipy.constants import Avogadro, atomic_mass, c, epsilon_0, k, physical_constants

from quantum_lab.engines import properties

WATER = "O 0 0 0;H 0 -.757 .587;H 0 .757 .587"


def reference(atoms=WATER, *, method="HF", functional="PBE", spin=0, charge=0):
    lib.num_threads(1)
    mol = gto.M(atom=atoms, unit="Angstrom", basis="sto-3g", spin=spin, charge=charge, verbose=0)
    if method == "DFT":
        mf = dft.UKS(mol) if spin else dft.RKS(mol)
        mf.xc = functional
        mf.grids.level = 1
    else:
        mf = scf.UHF(mol) if spin else scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 80
    mf.kernel()
    assert mf.converged
    return mf


@pytest.mark.science
@pytest.mark.parametrize("method,spin,atoms", [("HF", 0, WATER), ("DFT", 0, WATER),
                                               ("HF", 1, "H 0 0 0"), ("DFT", 1, "H 0 0 0")])
def test_populations_match_direct_mulliken_and_charge_sum(method, spin, atoms):
    mf = reference(atoms, method=method, spin=spin)
    answer = properties.analyze_properties(mf)
    _, direct_mulliken = mf.mulliken_pop(verbose=0)
    np.testing.assert_allclose(answer["mulliken_charges"], direct_mulliken, atol=1e-10)
    assert abs(sum(answer["mulliken_charges"]) - mf.mol.charge) < 1e-10
    assert abs(sum(answer["lowdin_charges"]) - mf.mol.charge) < 1e-10
    np.testing.assert_allclose(answer["dipole_debye"], mf.dip_moment(unit="Debye", verbose=0), atol=1e-10)
    assert answer["dipole_magnitude_debye"] == pytest.approx(np.linalg.norm(answer["dipole_debye"]))
    assert answer["spin_square"] == pytest.approx(0.75 if spin else 0)
    assert answer["reference_model"] == ({("HF", 0): "RHF", ("HF", 1): "UHF",
                                          ("DFT", 0): "RKS", ("DFT", 1): "UKS"}[(method, spin)])


@pytest.mark.science
def test_ion_populations_sum_to_charge_and_warn_about_dipole_origin():
    mf = reference("He 0 0 0", spin=1, charge=1)
    answer = properties.analyze_properties(mf)
    assert answer["mulliken_charge_sum"] == pytest.approx(1)
    assert answer["lowdin_charge_sum"] == pytest.approx(1)
    assert any("원점" in warning for warning in answer["warnings"])


@pytest.mark.science
def test_h2_frequency_matches_independent_energy_curvature_and_rrho(tmp_path):
    mf = reference("H 0 0 0;H 0 0 .712")
    result = properties.run_frequencies(mf, {"method": "HF", "conv_tol": 1e-11}, lambda *_: None, tmp_path)
    assert result["converged"] and result["stationary"]
    assert len(result["modes"]) == 1
    displacement = 0.001
    energies = []
    for sign in (-1, 1):
        coords = mf.mol.atom_coords()
        coords[1, 2] += sign * displacement
        mol = mf.mol.copy().set_geom_(coords, unit="Bohr")
        direct = scf.RHF(mol).run(conv_tol=1e-12)
        assert direct.converged
        energies.append(direct.e_tot)
    force_constant = (sum(energies) - 2 * mf.e_tot) / displacement**2
    reduced_mass_kg = mf.mol.atom_mass_list(isotope_avg=True)[0] / 2 * atomic_mass
    hartree_j = physical_constants["Hartree energy"][0]
    bohr_m = physical_constants["Bohr radius"][0]
    direct_frequency = np.sqrt(force_constant * hartree_j / (bohr_m**2 * reduced_mass_kg)) / (2 * np.pi * c * 100)
    assert result["frequencies_cm1"][0] == pytest.approx(direct_frequency, abs=0.02)
    thermal = result["thermochemistry"]
    assert thermal["gibbs_energy_hartree"] == pytest.approx(
        thermal["enthalpy_hartree"] - thermal["temperature_kelvin"] * thermal["entropy_hartree_per_kelvin"], abs=1e-11)
    # PySCF's NIST constants and current scipy.constants use different CODATA
    # editions; the ideal-gas identity agrees within their sub-ppm difference.
    assert thermal["enthalpy_hartree"] - thermal["internal_energy_hartree"] == pytest.approx(
        k * thermal["temperature_kelvin"] / hartree_j, rel=1e-6)
    assert result["zero_point_energy_hartree"] == pytest.approx(
        result["frequencies_cm1"][0] * properties.CM1_TO_HARTREE / 2, abs=1e-12)
    assert json.loads((tmp_path / "vibrations.json").read_text())["frequencies_cm1"] == result["frequencies_cm1"]
    with (tmp_path / "vibrations.csv").open() as handle:
        row = list(csv.DictReader(handle))[0]
    assert float(row["frequency_cm1"]) == result["frequencies_cm1"][0]
    assert row["ir_intensity_km_mol"] == ""


@pytest.mark.science
@pytest.mark.parametrize("functional", [None, "PBE", "PBE0", "B3LYP"])
def test_water_modes_match_direct_hessian_and_mass_orthogonality(functional, tmp_path):
    mf = reference(method="DFT" if functional else "HF", functional=functional or "PBE")
    spec = {"method": "DFT" if functional else "HF", "conv_tol": 1e-11, "max_cycle": 80}
    result = properties.run_frequencies(mf, spec, lambda *_: None, tmp_path)
    direct_hessian = mf.Hessian().kernel()
    direct = thermo.harmonic_analysis(mf.mol, direct_hessian, imaginary_freq=False)
    np.testing.assert_allclose(result["frequencies_cm1"], direct["freq_wavenumber"], atol=2e-4, rtol=0)
    assert len(result["modes"]) == 3
    modes = np.asarray([mode["displacements"] for mode in result["modes"]])
    masses = np.asarray(result["masses_amu"])
    np.testing.assert_allclose(np.einsum("max,nax,a->mn", modes, modes, masses), np.eye(3), atol=1e-10)
    # Each internal mode is orthogonal to rigid translation and rotation.
    np.testing.assert_allclose(np.einsum("max,a->mx", modes, masses), 0, atol=1e-10)
    coords = mf.mol.atom_coords()
    center = np.einsum("a,ax->x", masses, coords) / masses.sum()
    rotations = np.cross(coords[None, :, :] - center, modes)
    np.testing.assert_allclose(np.einsum("max,a->mx", rotations, masses), 0, atol=1e-10)
    assert not result["stationary"]
    assert result["thermochemistry"] is None
    assert result["zero_point_energy_hartree"] > 0
    assert not result["ir"]["computed"]
    assert all(mode["ir_intensity_km_mol"] is None for mode in result["modes"])


@pytest.mark.science
def test_stretched_h2_keeps_imaginary_mode_and_withholds_thermochemistry(tmp_path):
    mf = reference("H 0 0 0;H 0 0 2.5")
    result = properties.run_frequencies(mf, {"method": "HF"}, lambda *_: None, tmp_path)
    assert result["converged"]
    assert result["imaginary_modes"] == 1
    assert result["frequencies_cm1"][0] < 0
    assert result["modes"][0]["imaginary"]
    assert result["zero_point_energy_hartree"] is None
    assert result["thermochemistry"] is None


@pytest.mark.science
def test_homonuclear_h2_ir_is_zero_and_records_all_displacements(tmp_path):
    mf = reference("H 0 0 0;H 0 0 .712")
    result = properties.run_frequencies(mf, {"method": "HF", "ir_intensities": True}, lambda *_: None, tmp_path)
    assert result["converged"] and result["ir"]["computed"]
    assert result["ir"]["displaced_calculations"] == 12
    assert result["modes"][0]["ir_intensity_km_mol"] < 1e-14
    saved = json.loads((tmp_path / "ir-displacements.json").read_text())
    assert len(saved["calculations"]) == 12
    assert all(row["scf_converged"] for row in saved["calculations"])
    assert [row["displacement_bohr"] for row in saved["calculations"][:2]] == [-0.005, 0.005]


@pytest.mark.science
def test_water_ir_matches_independent_normal_coordinate_dipole_derivative(tmp_path):
    mf = reference()
    result = properties.run_frequencies(mf, {"method": "HF", "ir_intensities": True}, lambda *_: None, tmp_path)
    assert result["converged"] and result["ir"]["computed"]
    assert result["ir"]["displaced_calculations"] == 18
    mode = result["modes"][0]
    amplitude = 0.002
    dipoles = []
    for sign in (-1, 1):
        coordinates = mf.mol.atom_coords() + sign * amplitude * np.asarray(mode["displacements"])
        mol = mf.mol.copy().set_geom_(coordinates, unit="Bohr")
        direct = scf.RHF(mol).run(conv_tol=1e-12)
        assert direct.converged
        dipoles.append(np.asarray(direct.dip_moment(unit="AU", verbose=0)))
    derivative = (dipoles[1] - dipoles[0]) / (2 * amplitude)
    # A separate SI conversion uses a Debye/Å normal-coordinate derivative.
    debye_cm = 3.33564e-30
    atomic_dipole = physical_constants["atomic unit of electric dipole moment"][0]
    bohr_m = physical_constants["Bohr radius"][0]
    derivative_debye_angstrom = derivative * atomic_dipole / debye_cm / (bohr_m * 1e10)
    factor = Avogadro / (12 * epsilon_0 * c**2) * debye_cm**2 / (1e-20 * atomic_mass) / 1000
    direct_intensity = factor * np.dot(derivative_debye_angstrom, derivative_debye_angstrom)
    assert mode["ir_intensity_km_mol"] == pytest.approx(direct_intensity, rel=3e-4)


@pytest.mark.science
def test_rks_ir_preserves_parent_grid_and_geometry(tmp_path):
    mf = reference(method="DFT")
    original_grid = mf.grids.coords.copy()
    original_geometry = mf.mol.atom_coords().copy()
    result = properties.run_frequencies(mf, {"method": "DFT", "ir_intensities": True}, lambda *_: None, tmp_path)
    assert result["ir"]["computed"]
    assert all(mode["ir_intensity_km_mol"] >= 0 for mode in result["modes"])
    np.testing.assert_array_equal(mf.grids.coords, original_grid)
    np.testing.assert_array_equal(mf.mol.atom_coords(), original_geometry)


@pytest.mark.science
def test_ir_failure_keeps_modes_but_not_success_or_partial_intensities(monkeypatch, tmp_path):
    mf = reference()
    original = properties._displaced_reference

    def limited(reference, mol):
        displaced = original(reference, mol)
        displaced.max_cycle = 1
        return displaced

    monkeypatch.setattr(properties, "_displaced_reference", limited)
    result = properties.run_frequencies(mf, {"method": "HF", "ir_intensities": True}, lambda *_: None, tmp_path)
    assert not result["converged"]
    assert len(result["modes"]) == 3
    assert all(mode["ir_intensity_km_mol"] is None for mode in result["modes"])
    assert not result["ir"]["computed"]
    saved = json.loads((tmp_path / "ir-displacements.json").read_text())
    assert not saved["calculations"][-1]["scf_converged"]
    assert (tmp_path / "vibrations.npz").exists()


@pytest.mark.science
def test_unconverged_reference_cannot_produce_population_or_frequency_success(tmp_path):
    mf = scf.RHF(reference().mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 1
    mf.kernel()
    assert not mf.converged
    with pytest.raises(ValueError, match="converged"):
        properties.analyze_properties(mf)
    with pytest.raises(ValueError, match="converged"):
        properties.run_frequencies(mf, {}, lambda *_: None, tmp_path)


@pytest.mark.science
def test_frequency_adapter_preserves_result_artifacts_and_reference_density(tmp_path):
    from quantum_lab.engines.molecular import generate_volume, run

    spec = {"task": "frequencies", "method": "HF", "basis": "sto-3g", "restricted": True,
            "spin": 0, "charge": 0, "conv_tol": 1e-11, "threads": 1, "max_cycle": 80,
            "atoms": [{"symbol": "H", "position": [0, 0, 0]},
                      {"symbol": "H", "position": [0, 0, 0.712]}], "ir_intensities": True}
    events = []
    result = run(spec, lambda kind, data: events.append((kind, data)), tmp_path)
    assert result["converged"] and result["scf_converged"]
    assert result["vibrations"]["converged"] and result["vibrations"]["ir"]["computed"]
    assert result["properties"]["reference_model"] == "RHF"
    assert result["energy_hartree"] < -1.1
    assert any(kind == "stage" and data["name"] == "frequencies" for kind, data in events)
    for artifact in result["vibrations"]["artifacts"]:
        assert (tmp_path / artifact).exists()
    volume = generate_volume(tmp_path, {"kind": "density", "index": 0, "spin": 0, "grid": 20},
                             tmp_path / "density")
    assert volume["converged"]
    assert (tmp_path / "density" / volume["artifact"]).exists()


@pytest.mark.science
@pytest.mark.parametrize("method", ["HF", "DFT"])
def test_hessian_response_cycle_limit_cannot_produce_success(method, tmp_path):
    lib.num_threads(1)
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    mf = scf.RHF(mol) if method == "HF" else dft.RKS(mol)
    if method == "DFT":
        mf.xc, mf.grids.level = "PBE", 1
    mf.conv_tol, mf.max_cycle = 1e-12, 100
    mf.kernel()
    assert mf.converged
    original_krylov = lib.krylov
    with pytest.raises(RuntimeError, match="Hessian response failed"):
        properties.run_frequencies(mf, {"method": method, "max_cycle": 1, "conv_tol": 1e-12},
                                   lambda *_: None, tmp_path)
    assert lib.krylov is original_krylov
    assert not (tmp_path / "vibrations.json").exists()
    assert not json.loads((tmp_path / "hessian-response.json").read_text())["completed"]


@pytest.mark.science
def test_finite_unconverged_krylov_response_is_rejected(monkeypatch, tmp_path):
    mf = reference()
    original_krylov = lib.krylov

    def inaccurate(aop, b, *args, **kwargs):
        # Simulate the upstream warning-and-return case without replacing the
        # actual response operator, right-hand sides or Hessian construction.
        return np.zeros_like(b)

    inaccurate.__signature__ = properties.inspect.signature(original_krylov)
    monkeypatch.setattr(lib, "krylov", inaccurate)
    with pytest.raises(RuntimeError, match="CPSCF response residual"):
        properties.run_frequencies(mf, {"method": "HF"}, lambda *_: None, tmp_path)
    assert lib.krylov is inaccurate
    diagnostic = json.loads((tmp_path / "hessian-response.json").read_text())
    assert not diagnostic["completed"]
    assert diagnostic["checks"][0]["max_residual_norm"] > diagnostic["checks"][0]["requested_tolerance"]
