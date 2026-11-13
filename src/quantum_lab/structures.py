"""Import structures without inventing missing atoms, charges, or occupancies."""
import warnings
from io import StringIO
from typing import Literal

import numpy as np
from ase import Atoms
from ase.io.cif import parse_cif
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .models import Atom

router = APIRouter()


class StructureInput(BaseModel):
    format: Literal["xyz", "cif"]
    content: str = Field(min_length=1, max_length=1_000_000)


def parse_structure(fmt: str, content: str) -> dict:
    if fmt == "xyz":
        lines = content.strip().splitlines()
        if not lines:
            raise ValueError("원자 좌표를 입력하세요.")
        try:
            count = int(lines[0].strip())
        except ValueError:
            lines = [line.strip() for line in content.replace(";", "\n").splitlines() if line.strip()]
        else:
            if count < 1 or count > 200:
                raise ValueError("XYZ 원자 수는 1~200이어야 합니다.")
            if len(lines) != count + 2:
                raise ValueError("XYZ 원자 수와 좌표 줄 수가 일치하지 않습니다. 한 구조와 주석 한 줄만 입력하세요.")
            lines = lines[2:]
        parsed = []
        for line in lines:
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("좌표의 각 줄은 '원소 x y z' 네 항목이어야 합니다.")
            try:
                parsed.append(Atom(symbol=fields[0], position=tuple(float(x) for x in fields[1:])))
            except (TypeError, ValueError) as exc:
                raise ValueError("XYZ 원소와 유한한 좌표 세 값을 확인하세요.") from exc
        atoms = Atoms(symbols=[a.symbol for a in parsed], positions=[a.position for a in parsed])
        lattice = None
    elif fmt == "cif":
        with warnings.catch_warnings(record=True) as caught:
            try:
                blocks = [block for block in parse_cif(StringIO(content)) if block.has_structure()]
                if len(blocks) != 1:
                    raise ValueError("원자 구조가 정확히 하나 포함된 CIF를 입력하세요.")
                block = blocks[0]
                # Inspect raw site occupancies before ASE expands symmetry or
                # combines coincident sites into a dominant-species structure.
                occupancies = block.get("_atom_site_occupancy")
                if occupancies is not None:
                    values = np.asarray(occupancies, dtype=float)
                    if not np.isfinite(values).all() or np.any(np.abs(values - 1) > 1e-8):
                        raise ValueError("부분 점유·무질서 CIF는 지원하지 않습니다. 완전 점유 구조를 준비하세요.")
                for key in ("_atom_site_disorder_group", "_atom_site_disorder_assembly"):
                    if any(str(value).strip() not in {".", "?", "0"} for value in block.get(key, [])):
                        raise ValueError("무질서 그룹을 포함한 CIF는 지원하지 않습니다.")
                atoms = block.get_atoms(fractional_occupancies=True)
            except (IndexError, KeyError, TypeError, AssertionError, StopIteration) as exc:
                raise ValueError("CIF의 격자·원소·원자 좌표를 해석할 수 없습니다.") from exc
        occupancy = atoms.info.get("occupancy", {})
        for components in occupancy.values():
            if len(components) != 1 or any(abs(float(x) - 1.0) > 1e-8 for x in components.values()):
                raise ValueError("부분 점유·무질서 CIF는 지원하지 않습니다. 완전 점유 구조를 준비하세요.")
        if any("occupanc" in str(w.message).lower() for w in caught):
            raise ValueError("CIF 점유 정보를 명확히 해석할 수 없습니다.")
        if atoms.cell.rank != 3 or not all(atoms.pbc):
            raise ValueError("3차원 주기 격자가 필요합니다.")
        lattice = atoms.cell.array.tolist()
    else:
        raise ValueError("XYZ 또는 CIF만 지원합니다.")
    if not 1 <= len(atoms) <= 200 or not np.isfinite(atoms.positions).all():
        raise ValueError("1~200개의 유효한 원자 좌표가 필요합니다.")
    parsed_atoms = [Atom(symbol=a.symbol, position=tuple(a.position)).model_dump(mode="json") for a in atoms]
    return {"atoms": parsed_atoms,
            "lattice": lattice, "domain": "crystal" if lattice else "molecule"}


@router.post("/structures/parse")
def parse(body: StructureInput):
    try:
        return parse_structure(body.format, body.content)
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
