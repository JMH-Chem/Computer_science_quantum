"""Validated public inputs; all coordinates are Angstrom, spin is Nα − Nβ."""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Literal
from uuid import UUID

import numpy as np
from ase.data import atomic_numbers
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


@lru_cache(maxsize=256)
def _check_functional(name: str) -> None:
    from pyscf.dft import libxc

    try:
        _, components = libxc.parse_xc(name)
        kinetic_ids = {int(code) for label, code in libxc.XC_CODES.items()
                       if "_K_" in label and isinstance(code, Integral)}
        if any(int(code) in kinetic_ids for code, _ in components):
            raise ValueError("운동에너지 함수형은 KS 교환·상관 함수형으로 사용할 수 없습니다.")
        if libxc.needs_laplacian(name):
            raise ValueError("밀도 Laplacian 의존 함수형은 현재 어댑터에서 지원하지 않습니다.")
        if not libxc.test_deriv_order(name, 2, raise_error=False):
            raise ValueError("현재 어댑터에 필요한 함수형 도함수가 지원되지 않습니다.")
    except (KeyError, RuntimeError, NotImplementedError) as exc:
        raise ValueError(f"지원하지 않거나 해석할 수 없는 함수형: {name}") from exc


class Atom(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    symbol: str
    position: tuple[float, float, float]

    @field_validator("symbol")
    @classmethod
    def element(cls, value: str) -> str:
        value = value.strip().capitalize()
        if value not in atomic_numbers or value == "X":
            raise ValueError("실제 원소 기호를 입력하세요. Ghost atom은 첫 버전에서 지원하지 않습니다.")
        return value


class VolumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["density", "orbital", "nto_hole", "nto_particle"] = "density"
    index: int = Field(default=0, ge=0, le=1000)
    spin: int = Field(default=0, ge=0, le=1)
    grid: int = Field(default=40, ge=20, le=100)


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(default="새 계산", min_length=1, max_length=120)
    domain: Literal["molecule", "crystal"] = "molecule"
    task: Literal["single_point", "optimize", "tda", "tddft", "bands", "educational_rhf", "volume"] = "single_point"
    atoms: list[Atom] = Field(min_length=1, max_length=200)
    lattice: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None
    charge: int = Field(default=0, ge=-20, le=20)
    spin: int = Field(default=0, ge=0, le=100)
    method: Literal["HF", "DFT"] = "HF"
    basis: str | dict[str, str] = "sto-3g"
    ecp: str | dict[str, str] | None = None
    functional: str = Field(default="PBE", min_length=1, max_length=160)
    restricted: bool = True
    kmesh: tuple[int, int, int] = (1, 1, 1)
    pseudo: str = "gth-pbe"
    grid_level: int = Field(default=2, ge=0, le=6)
    mesh: tuple[int, int, int] = (15, 15, 15)
    max_cycle: int = Field(default=80, ge=1, le=500)
    conv_tol: float = Field(default=1e-9, ge=1e-12, le=1e-3)
    nstates: int = Field(default=5, ge=1, le=20)
    max_steps: int = Field(default=50, ge=1, le=200)
    threads: int = Field(default=2, ge=1, le=8)
    memory_mb: int = Field(default=2000, ge=256, le=16000)
    band_points: int = Field(default=40, ge=5, le=200)
    parent_job_id: str | None = None
    volume: VolumeRequest | None = None

    @field_validator("basis", "ecp")
    @classmethod
    def named_basis(cls, value):
        if value is None:
            return value
        entries = value.items() if isinstance(value, dict) else [(None, value)]
        normalized = {}
        for symbol, name in entries:
            if symbol is not None:
                symbol = symbol.strip().capitalize()
                if symbol not in atomic_numbers or symbol == "X":
                    raise ValueError(f"알 수 없는 원소: {symbol}")
                if symbol in normalized:
                    raise ValueError(f"중복된 원소 설정: {symbol}")
            if not isinstance(name, str) or not name.strip() or len(name) > 160:
                raise ValueError("기저/ECP는 등록된 이름을 지정하세요.")
            normalized[symbol] = name.strip()
        if isinstance(value, dict) and not value:
            raise ValueError("빈 기저/ECP 사전은 사용할 수 없습니다.")
        return normalized if isinstance(value, dict) else value.strip()

    @field_validator("kmesh")
    @classmethod
    def kmesh_bounds(cls, value):
        if any(x < 1 or x > 4 for x in value):
            raise ValueError("첫 버전의 k점 격자는 축마다 1~4입니다.")
        return value

    @field_validator("mesh")
    @classmethod
    def mesh_bounds(cls, value):
        if any(x < 9 or x > 99 for x in value):
            raise ValueError("FFT 격자는 축마다 9~99입니다.")
        return value

    @model_validator(mode="after")
    def compatible(self):
        elements = {a.symbol for a in self.atoms}
        if isinstance(self.basis, dict) and elements - self.basis.keys():
            raise ValueError("모든 원소에 대해 기저를 지정하세요.")
        for definitions in (self.basis, self.ecp):
            if isinstance(definitions, dict) and definitions.keys() - elements:
                raise ValueError("구조에 없는 원소의 기저/ECP가 지정되었습니다.")
        electrons = sum(atomic_numbers[a.symbol] for a in self.atoms) - self.charge
        if self.domain == "molecule" and self.ecp:
            import basis_set_exchange as bse

            core_electrons = {}
            for symbol in elements:
                name = self.ecp.get(symbol) if isinstance(self.ecp, dict) else self.ecp
                if not name:
                    core_electrons[symbol] = 0
                    continue
                try:
                    definition = bse.get_basis(name, elements=[symbol])
                    core_electrons[symbol] = int(
                        definition["elements"][str(atomic_numbers[symbol])].get("ecp_electrons", 0)
                    )
                except (KeyError, ValueError, RuntimeError) as exc:
                    raise ValueError(f"{symbol}의 ECP 정의를 읽을 수 없습니다: {name}") from exc
            electrons -= sum(core_electrons[a.symbol] for a in self.atoms)
        if electrons < 1 or electrons < self.spin or (electrons - self.spin) % 2:
            raise ValueError("전자 수와 spin=Nα−Nβ의 홀짝 또는 크기가 맞지 않습니다.")
        if self.restricted and self.spin != 0:
            raise ValueError("spin≠0이면 비제한(UHF/UKS) 계산을 선택하세요.")
        if self.method == "DFT":
            _check_functional(self.functional)
        coords = np.array([a.position for a in self.atoms])
        if not np.isfinite(coords).all():
            raise ValueError("좌표는 유한한 수여야 합니다.")
        for i, p in enumerate(coords):
            if any(np.linalg.norm(p - q) < 1e-5 for q in coords[:i]):
                raise ValueError("두 원자가 같은 위치에 있습니다.")
        if self.domain == "crystal":
            if self.lattice is None or not np.isfinite(self.lattice).all() or abs(np.linalg.det(self.lattice)) < 1e-6:
                raise ValueError("독립적인 세 격자 벡터가 필요합니다.")
            if self.charge or self.spin or not self.restricted:
                raise ValueError("첫 버전의 주기계는 중성 닫힌껍질만 지원합니다.")
            if self.task not in {"single_point", "bands", "volume"}:
                raise ValueError("주기계는 고정 구조 DFT와 밴드 계산을 지원합니다.")
            if self.method != "DFT" or self.functional.upper() != "PBE" or self.pseudo.lower() != "gth-pbe":
                raise ValueError("주기계 검증 경로는 PBE / GTH-PBE입니다.")
            if not isinstance(self.basis, str) or self.basis.lower() not in {"gth-szv", "gth-dzvp"} or self.ecp:
                raise ValueError("주기계에는 GTH-SZV 또는 GTH-DZVP와 GTH-PBE를 사용하세요.")
            fractional = np.linalg.solve(np.asarray(self.lattice).T, coords.T).T
            for i, position in enumerate(fractional):
                displacement = fractional[:i] - position
                displacement -= np.rint(displacement)
                if np.any(np.linalg.norm(displacement @ self.lattice, axis=1) < 1e-5):
                    raise ValueError("주기 경계를 적용하면 두 원자가 같은 위치에 있습니다.")
            if self.task == "volume" and self.volume and self.volume.kind != "density":
                raise ValueError("첫 버전의 주기계는 가전자 밀도 격자만 지원합니다.")
        else:
            if self.lattice is not None:
                raise ValueError("분자 계산에는 격자 벡터를 지정하지 않습니다.")
            if self.task == "bands":
                raise ValueError("밴드는 주기계 계산입니다.")
            if self.kmesh != (1, 1, 1):
                raise ValueError("분자 계산에는 k점 격자를 적용하지 않습니다.")
        if self.task in {"tda", "tddft"}:
            if self.spin or not self.restricted or self.method != "DFT":
                raise ValueError("여기 상태는 닫힌껍질 RKS singlet TDA/TDDFT를 지원합니다.")
            if self.functional.upper() not in {"PBE", "PBE0", "B3LYP"}:
                raise ValueError("여기 상태 검증 경로는 PBE, PBE0, B3LYP입니다.")
        if self.task == "optimize" and self.method == "DFT" and self.functional.upper() not in {"PBE", "PBE0", "B3LYP"}:
            raise ValueError("DFT 최적화 검증 경로는 PBE, PBE0, B3LYP입니다.")
        if self.task == "educational_rhf" and (self.method != "HF" or self.spin or not self.restricted or self.ecp):
            raise ValueError("교육용 RHF는 ECP 없는 닫힌껍질 HF 계산입니다.")
        if self.task == "volume":
            if not self.parent_job_id or self.volume is None:
                raise ValueError("격자 생성에는 원본 작업과 요청이 필요합니다.")
            UUID(self.parent_job_id)
        elif self.parent_job_id is not None or self.volume is not None:
            raise ValueError("원본 작업 지정은 격자 생성에서만 사용합니다.")
        return self
