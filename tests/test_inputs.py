"""Structure, physical input and capability-catalog boundary checks."""

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from quantum_lab.catalog import functional_catalog, search_bases
from quantum_lab.models import JobSpec
from quantum_lab.structures import parse_structure

H2 = [{"symbol": "H", "position": [0, 0, 0]}, {"symbol": "H", "position": [0, 0, 0.74]}]


def cif(occupancy="1", atom="C"):
    return f"""data_sample
_cell_length_a 3.5
_cell_length_b 3.5
_cell_length_c 3.5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
{atom}1 {atom} 0 0 0 {occupancy}
"""


@pytest.mark.parametrize("text", [
    "2\nHydrogen\nH 0 0 0\nH 0 0 .74\n",
    "2\n\nH 0 0 0\nH 0 0 .74\n",
    "H 0 0 0\nH 0 0 .74",
    "h 0 0 0; H 0 0 7.4e-1",
])
def test_xyz_counted_and_raw_coordinates(text):
    result = parse_structure("xyz", text)
    assert result["domain"] == "molecule"
    assert result["lattice"] is None
    assert result["atoms"] == H2


@pytest.mark.parametrize("text", [
    " ", "0\nempty\n", "2\ncomment\nH 0 0 0\n",  # missing atom
    "1\ncomment\nH 0 0 0\nH 0 0 1\n",  # surplus atom
    "1\nframe 1\nH 0 0 0\n1\nframe 2\nH 0 0 1\n",  # no silent first-frame import
    "H nan 0 0", "H 0 inf 0", "Unknown 0 0 0", "X 0 0 0",
    "H 0 0 0 ignored-column",
])
def test_xyz_rejects_missing_ambiguous_and_nonfinite_data(text):
    with pytest.raises(ValueError):
        parse_structure("xyz", text)


def test_cif_full_occupancy_and_cell_units():
    result = parse_structure("cif", cif())
    assert result["domain"] == "crystal"
    assert result["atoms"][0]["symbol"] == "C"
    np.testing.assert_allclose(result["lattice"], np.eye(3) * 3.5, atol=1e-12)


@pytest.mark.parametrize("occupancy", ["0.5", "0.5(1)", "0", "1.1", "nan", "?"])
def test_cif_partial_missing_or_invalid_occupancy_is_not_normalized(occupancy):
    with pytest.raises(ValueError):
        parse_structure("cif", cif(occupancy))


def test_cif_disorder_and_multiple_structures_are_explicit_errors():
    disorder = cif().replace("_atom_site_occupancy", "_atom_site_occupancy\n_atom_site_disorder_group")
    disorder = disorder.replace("C1 C 0 0 0 1", "C1 C 0 0 0 1 1")
    with pytest.raises(ValueError, match="무질서"):
        parse_structure("cif", disorder)
    with pytest.raises(ValueError, match="하나"):
        parse_structure("cif", cif() + cif().replace("data_sample", "data_second"))
    with pytest.raises(ValueError):
        parse_structure("cif", "data_broken\n_cell_length_a 3\n")


def test_spin_parity_and_unrestricted_open_shell():
    with pytest.raises(ValidationError, match="홀짝"):
        JobSpec(atoms=H2, spin=1, restricted=False)
    atom = [{"symbol": "H", "position": [0, 0, 0]}]
    with pytest.raises(ValidationError, match="비제한"):
        JobSpec(atoms=atom, spin=1, restricted=True)
    valid = JobSpec(atoms=atom, spin=1, restricted=False)
    assert valid.spin == 1
    assert JobSpec(atoms=H2, charge=1, spin=1, restricted=False).charge == 1


def test_spin_cannot_exceed_ecp_valence_electrons():
    atom = [{"symbol": "Na", "position": [0, 0, 0]}]
    with pytest.raises(ValidationError, match="크기"):
        JobSpec(atoms=atom, basis="lanl2dz", ecp="lanl2dz", spin=3, restricted=False)
    assert JobSpec(atoms=atom, basis="lanl2dz", ecp="lanl2dz", spin=1, restricted=False).spin == 1


def test_element_basis_assignment_normalization_and_coverage():
    assert JobSpec(atoms=H2, basis={"h": " sto-3g "}).basis == {"H": "sto-3g"}
    for basis in ({}, {"He": "sto-3g"}, {"H": "sto-3g", "Xe": "sto-3g"},
                  {"X": "sto-3g"}, {"h": "sto-3g", "H": "def2-svp"}):
        with pytest.raises(ValidationError):
            JobSpec(atoms=H2, basis=basis)


@pytest.mark.parametrize("update", [
    {"task": "bands"}, {"lattice": np.eye(3).tolist()}, {"kmesh": [2, 1, 1]},
    {"task": "tddft", "method": "HF"},
    {"task": "tddft", "method": "DFT", "restricted": False},
    {"method": "DFT", "functional": "not_a_functional"},
    {"method": "DFT", "functional": "GGA_K_TFVW"},
    {"task": "volume"},
])
def test_invalid_molecular_task_settings_rejected_before_job(update):
    with pytest.raises(ValidationError):
        JobSpec(atoms=H2, **update)


def crystal_spec():
    return {
        "domain": "crystal", "method": "DFT", "functional": "PBE", "basis": "gth-szv",
        "atoms": [{"symbol": "C", "position": [0, 0, 0]},
                  {"symbol": "C", "position": [1, 1, 1]}],
        "lattice": (np.eye(3) * 3.5).tolist(),
    }


def test_crystal_model_and_periodically_duplicate_atoms():
    assert JobSpec(**crystal_spec()).domain == "crystal"
    for update in ({"charge": 2}, {"spin": 2, "restricted": False}, {"method": "HF"},
                   {"task": "optimize"}, {"basis": "sto-3g"}, {"functional": "B3LYP"},
                   {"lattice": [[0, 0, 0]] * 3}):
        with pytest.raises(ValidationError):
            JobSpec(**(crystal_spec() | update))
    spec = crystal_spec()
    spec["atoms"][1]["position"] = [3.5, 0, 0]
    with pytest.raises(ValidationError, match="주기 경계"):
        JobSpec(**spec)
    with pytest.raises(ValidationError, match="밀도"):
        JobSpec(**(crystal_spec() | {"task": "volume", "parent_job_id": "00000000-0000-0000-0000-000000000001",
                                   "volume": {"kind": "orbital"}}))


def test_catalog_includes_numpy_integer_libxc_codes_and_hybrid_families():
    entries = {entry["name"]: entry for entry in functional_catalog()}
    assert len(entries) > 500  # installed Libxc provides hundreds of integer-code entries
    assert entries["GGA_X_PBE"]["component"] == "exchange"
    assert entries["PBE"]["family"] == "GGA"
    assert entries["PBE0"]["family"] == "hybrid GGA"
    assert entries["B3LYP"]["hybrid"] is True
    assert entries["HYB_GGA_XC_CAM_B3LYP"]["range_separated"] is True
    assert entries["GGA_K_TFVW"]["status"] == "unsupported"
    assert entries["GGA_K_TFVW"]["supported_tasks"] == []
    for entry in entries.values():
        assert entry["verification"] == "combination_not_asserted"
        if entry["status"] == "unsupported":
            assert entry["supported_tasks"] == []
    assert entries["PBE0"]["references"]


def test_basis_search_respects_elements_and_role():
    matches = search_bases(q="def2-svp", elements=" h, o ")
    assert matches
    assert all({"1", "8"} <= set(row["elements"]) for row in matches)
    assert all(row["role"] == "orbital" for row in matches)
    assert not search_bases(q="def2-universal-jkfit", elements="C")
    with pytest.raises(HTTPException) as error:
        search_bases(q="", elements="C,not_an_element")
    assert error.value.status_code == 422
