"""Frozen controlled experiments and conservative, data-derived interpretation.

Only final converged worker results are compared. An energy difference is not
an accuracy certificate, and no failed sample is replaced or interpolated.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from .examples import examples
from .models import JobSpec
from .storage import TERMINAL

TEMPLATE_ID = "water_basis_v1"
BASES = ("sto-3g", "def2-svp", "def2-tzvp")
INTERPRETATION_VERSION = "2"
PROVENANCE_KEYS = ("basis_provider", "basis_provider_version", "libxc_version", "numpy_version")


def _configs() -> list[dict]:
    """Definitions are versioned; complete inputs and wording are frozen at creation."""
    return [
        dict(id=TEMPLATE_ID, example="water", variable="basis", variable_label="기저 함수 집합",
             values=list(BASES), labels=list(BASES), title="물의 기저를 바꾸면 무엇이 달라질까?",
             question="전자 오비탈을 표현하는 함수를 늘리면 에너지와 계산 비용은 어떻게 달라질까요?",
             description="같은 물 분자를 PBE로 세 번 계산하고, 기저만 바꾸어 결과를 비교합니다.",
             limit="STO-3G는 최소 기저이며 def2 계열과 체계적으로 중첩된 한 계열이 아닙니다. 수치 DFT 적분에서 에너지가 항상 단조롭게 낮아진다고 보장할 수 없습니다.",
             next="각 계산을 열어 SCF 이력과 밀도를 확인하세요. 다음에는 같은 기저를 고정하고 적분 격자를 바꾸어 관심 물성의 수렴을 따로 확인할 수 있습니다."),
        dict(id="water_grid_v1", example="water", variable="grid_level", variable_label="DFT 적분 격자 수준",
             values=[1, 3, 5], labels=["Level 1", "Level 3", "Level 5"],
             title="물의 적분 격자를 촘촘하게 하면?", question="기저를 고정하고 적분 격자를 바꾸면 에너지 차이는 얼마나 줄어들까요?",
             description="PBE/def2-SVP 물 분자의 수치 적분 격자만 1, 3, 5 수준으로 바꿉니다.",
             limit="DFT 적분 격자는 교환·상관 적분의 수치 정확도를 조절합니다. 격자 수준은 기저 함수 수가 아니며 에너지 변화가 반드시 단조롭지는 않습니다. 세 점만으로 모든 물성의 수렴을 인증하지 않습니다.",
             next="에너지뿐 아니라 관심 물성에 필요한 허용오차를 정하고, 더 큰 격자와 기저의 영향을 각각 검토하세요."),
        dict(id="water_functional_v1", example="water", variable="functional", variable_label="교환·상관 함수형",
             values=["PBE", "PBE0", "B3LYP"], labels=["PBE", "PBE0", "B3LYP"],
             title="함수형이 다르면 무엇이 달라질까?", question="구조와 기저가 같아도 교환·상관 근사가 다르면 어떤 결과를 얻을까요?",
             description="같은 물 구조, def2-SVP, 적분 격자 3에서 PBE·PBE0·B3LYP를 비교합니다.",
             limit="서로 다른 함수형은 서로 다른 에너지 근사입니다. 전체 에너지가 더 낮다는 이유로 더 정확하거나 우수한 함수형이라고 순위를 매길 수 없습니다. 관심 물성을 실험 또는 적절한 기준 계산과 비교해야 합니다.",
             next="함수형마다 dipole과 궤도 간격도 비교하고, 비교할 물리량과 신뢰할 기준값을 먼저 정하세요."),
        dict(id="hydrogen_dissociation_v1", example="h2", variable="bond_distance", variable_label="H–H 거리 / Å",
             values=[0.5, 0.74, 1.0, 1.5, 2.0], labels=["0.50 Å", "0.74 Å", "1.00 Å", "1.50 Å", "2.00 Å"],
             title="수소 분자의 결합을 늘려 보면?", question="H–H 거리를 늘리면 전체 에너지는 어떻게 달라질까요?",
             description="RHF/STO-3G를 고정하고 H–H 거리 5개에서 독립적인 단일점 계산을 실행합니다.",
             limit="이 곡선은 정해진 거리에서 계산한 점들의 연결이며 실제 시간 운동이나 구조 최적화가 아닙니다. 멀리 떨어진 H₂는 다중참조 성격이 커져 제한 HF의 해리 극한이 잘못될 수 있습니다.",
             next="최저 에너지 표본이 정확한 평형 결합 길이는 아닙니다. 최소점 주변을 더 촘촘히 계산하거나 구조 최적화와 상관 방법을 비교하세요."),
        dict(id="diamond_kmesh_v1", example="diamond", variable="kmesh", variable_label="k점 격자",
             values=[[1, 1, 1], [2, 2, 2], [3, 3, 3]], labels=["Γ · 1×1×1", "2×2×2", "3×3×3"],
             title="결정의 k점을 늘리면?", question="Brillouin zone을 더 많은 k점으로 적분하면 단위격자 에너지는 어떻게 달라질까요?",
             description="다이아몬드 PBE/GTH-SZV/GTH-PBE, FFT 15³를 고정하고 k점 격자만 늘립니다.",
             limit="FFT 15³와 작은 기저는 빠른 학습용 설정입니다. k점 변화만으로 FFT 격자나 기저의 오차가 제거되지는 않습니다. 계산한 k점의 KS 간격은 광학 갭 또는 전체 Brillouin zone의 엄밀한 갭이 아닙니다.",
             next="원자당 에너지 차이를 확인한 뒤 FFT 격자와 기저를 각각 바꿔 수렴성을 검토하세요."),
        dict(id="diamond_mesh_v1", example="diamond", variable="mesh", variable_label="FFT 격자",
             values=[[15, 15, 15], [21, 21, 21], [27, 27, 27]], labels=["15×15×15", "21×21×21", "27×27×27"],
             title="결정의 FFT 격자를 바꾸면?", question="같은 Γ점과 궤도 기저에서 보조 수치 격자를 늘리면 에너지는 얼마나 달라질까요?",
             description="다이아몬드 PBE/GTH-SZV/GTH-PBE, Γ점을 고정하고 FFT 격자만 바꿉니다.",
             limit="Gaussian 궤도 기저의 FFT 보조 격자 비교입니다. 평면파 궤도 기저의 cutoff 비교가 아니며, Γ점만 사용했기 때문에 k점 수렴을 주장할 수 없습니다.",
             next="같은 FFT 격자에서 k점과 기저를 각각 바꾸고, 관심 물성의 수렴 허용오차를 별도로 정하세요."),
        dict(id="water_correlation_v1", example="water", variable="method", variable_label="전자 구조 방법",
             values=["HF", "MP2", "CCSD"], labels=["HF", "MP2", "CCSD"],
             title="전자 상관을 포함하면?", question="같은 물 구조와 작은 기저에서 평균장과 상관 방법의 에너지는 어떻게 달라질까요?",
             description="물의 STO-3G 기저와 구조를 고정하고 RHF·MP2·CCSD 단일점 에너지를 비교합니다.",
             limit="STO-3G는 교육용 최소 기저입니다. MP2와 CCSD는 보편적인 변분 상한을 제공하지 않으며, 낮은 에너지가 정확도 순위는 아닙니다. 표시되는 궤도와 밀도는 상관 방법의 RHF 참조입니다.",
             next="참조 에너지와 상관 에너지를 구분하고, 더 큰 기저와 단일참조 근사의 적합성을 별도로 확인하세요."),
    ]


def _config(template_id: str) -> dict:
    for config in _configs():
        if config["id"] == template_id:
            return config
    raise ValueError("지원하지 않는 학습 실험입니다.")


def build_specs(template_id: str, name: str) -> list[dict]:
    config = _config(template_id)
    base = deepcopy(next(example["spec"] for example in examples() if example["id"] == config["example"]))
    base.update(grid_level=3, threads=1, conv_tol=1e-9, max_cycle=80, task="single_point")
    if config["variable"] == "method":
        base.update(basis="sto-3g", method="HF", restricted=True, spin=0)
    specs = []
    for value, label in zip(config["values"], config["labels"], strict=True):
        spec = deepcopy(base)
        if config["variable"] == "bond_distance":
            spec["atoms"][1]["position"] = [0, 0, value]
        else:
            spec[config["variable"]] = deepcopy(value)
        spec["name"] = f"{name[:85]} · {label}"
        specs.append(JobSpec.model_validate(spec).model_dump(mode="json"))
    return specs


def _presentation(config: dict, specs: list[dict]) -> dict:
    variable, domain = config["variable"], specs[0]["domain"]
    fixed = {"basis": "구조·전하·스핀·PBE·수치 격자·SCF 기준",
             "grid_level": "구조·기저·PBE·SCF 기준", "functional": "구조·기저·적분 격자·SCF 기준",
             "bond_distance": "원소·전하·스핀·RHF·기저·SCF 기준", "kmesh": "구조·PBE·기저·가상퍼텐셜·FFT 격자·SCF 기준",
             "mesh": "구조·PBE·기저·가상퍼텐셜·k점·SCF 기준", "method": "구조·STO-3G 기저·전하·스핀·수렴 기준"}[variable]
    return {
        "id": config["id"], "template_version": "1", "domain": domain,
        "title": config["title"], "question": config["question"], "description": config["description"],
        "variable": variable, "variable_label": config["variable_label"],
        "values": deepcopy(config["values"]), "labels": config["labels"].copy(),
        "bases": [deepcopy(spec["basis"]) for spec in specs], "fixed_spec": deepcopy(specs[0]),
        "predictions": [
            {"id": "lower", "label": "에너지가 낮아질 것 같아요"},
            {"id": "higher", "label": "에너지가 높아질 것 같아요"},
            {"id": "unsure", "label": "아직 모르겠어요"},
        ],
        "lessons": [
            {"title": "01 · 예측하기", "body": config["question"] + " 예상은 계산값을 바꾸지 않습니다."},
            {"title": "02 · 한 가지 조건만 바꾸기", "body": f"{fixed}을 고정하고 {config['variable_label']}만 바꿉니다. 한 번에 한 작업씩 CPU에서 실행합니다."},
            {"title": "03 · 관찰하기", "body": "최종 에너지와 바로 이전 표본과의 부호 있는 ΔE, 계산 시간과 프로세스 최대 메모리를 확인하세요. 표의 계산을 열면 실제 수렴 이력을 볼 수 있습니다."},
            {"title": "04 · 해석하기", "body": config["limit"]},
        ],
        "units": {"coordinates": "angstrom", "energy": "hartree", "delta_energy": "hartree",
                  "energy_per_atom": "hartree/atom", "elapsed_seconds": "s", "peak_memory_mb": "MiB"},
        "references": ["https://pyscf.org/user/pbc/scf.html"] if domain == "crystal" else
                      ["https://www.basissetexchange.org/", "https://pyscf.org/user/dft.html",
                       "https://pyscf.org/user/post_hf.html"],
        "limit": config["limit"], "next": config["next"],
    }


def templates() -> list[dict]:
    return [_presentation(config, build_specs(config["id"], config["title"])) for config in _configs()]


def freeze_definition(template_id: str, specs: list[dict]) -> dict:
    """Called once inside experiment creation, before jobs become runnable."""
    config = _config(template_id)
    if len(specs) != len(config["values"]):
        raise ValueError("실험의 조건 수가 템플릿과 일치하지 않습니다.")
    return {**_presentation(config, specs), "expected_specs": deepcopy(specs)}


def _legacy_definition(record: dict) -> dict:
    """v0.2 records have only basis variations; never fill from current defaults."""
    specs = [{**deepcopy(record["base_spec"]), "basis": deepcopy(basis)} for basis in record["bases"]]
    config = _config(TEMPLATE_ID)
    config.update(values=deepcopy(record["bases"]), labels=[str(b) for b in record["bases"]])
    return {**_presentation(config, specs), "expected_specs": specs}


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) else None


def _positive(value, *, integer=False):
    number = _finite(value)
    if number is None or number < 0 or (integer and (number != int(number) or number == 0)):
        return None
    return int(number) if integer else number


def _input_without_name(spec: dict):
    return {key: value for key, value in spec.items() if key != "name"}


def _geometry_matches(reference: list, observed: list) -> bool:
    if len(reference) != len(observed) or not reference:
        return False
    if [a.get("symbol") for a in reference] != [a.get("symbol") for a in observed]:
        return False
    try:
        a = np.asarray([atom["position"] for atom in reference], dtype=float)
        b = np.asarray([atom["position"] for atom in observed], dtype=float)
        return a.shape == b.shape and bool(np.isfinite(b).all() and np.allclose(a, b, rtol=0, atol=1e-8))
    except (KeyError, TypeError, ValueError):
        return False


def _signature(result: dict) -> tuple | None:
    engine, provenance = result.get("engine") or {}, result.get("provenance") or {}
    values = (engine.get("name"), engine.get("version"), *(provenance.get(key) for key in PROVENANCE_KEYS))
    return values if all(isinstance(value, str) and value for value in values) else None


def summarize(experiment: dict, jobs: list[dict]) -> dict:
    """Report exactly the persisted ordered inputs; failed neighbors leave gaps."""
    record = {key: deepcopy(value) for key, value in experiment.items() if key != "request_payload"}
    definition = record.get("definition") or _legacy_definition(record)
    expected_specs = definition["expected_specs"]
    labels, values = definition["labels"], definition["values"]
    rows, signatures = [], set()
    for index, expected in enumerate(expected_specs):
        job = jobs[index] if index < len(jobs) else {}
        result, spec = job.get("result") or {}, job.get("spec") or {}
        status = job.get("status", "interrupted")
        reason = None
        energy = _finite(result.get("energy_hartree"))
        if status != "succeeded" or result.get("converged") is not True or result.get("scf_converged") is not True:
            reason = "요청한 방법과 최종 SCF 수렴을 확인한 계산만 비교합니다."
        elif energy is None:
            reason = "유한한 최종 에너지가 없습니다."
        elif _input_without_name(spec) != _input_without_name(expected):
            reason = "저장한 실험 조건 또는 표본 순서와 달라 비교에서 제외했습니다."
        elif (result.get("method") != spec.get("method") or result.get("task") != spec.get("task")
              or result.get("lattice") != spec.get("lattice")
              or not _geometry_matches(expected.get("atoms", []), result.get("atoms") or [])):
            reason = "결과의 구조·방법과 저장한 입력 조건이 일치하지 않습니다."
        elif _signature(result) is None:
            reason = "엔진·기저 제공자·라이브러리 버전 기록을 확인할 수 없습니다."
        if reason is None:
            signatures.add(_signature(result))
        count = len(expected["atoms"])
        rows.append({
            "ordinal": index, "job_id": job.get("id"), "basis": expected["basis"],
            "label": labels[index], "value": deepcopy(values[index]), "status": status,
            "comparable": reason is None, "energy_hartree": energy if reason is None else None,
            "delta_previous_hartree": None,
            "energy_per_atom_hartree": energy / count if reason is None and count else None,
            "delta_previous_per_atom_hartree": None,
            "basis_functions": _positive(result.get("basis_functions"), integer=True),
            "iteration_count": _positive(result.get("iteration_count"), integer=True),
            "elapsed_seconds": _positive(result.get("elapsed_seconds")),
            "peak_memory_mb": _positive(result.get("peak_memory_mb")),
            "error": job.get("error"), "comparison_note": reason,
        })

    if len(signatures) > 1:
        for row in rows:
            row.update(comparable=False, energy_hartree=None, energy_per_atom_hartree=None,
                       comparison_note="엔진 또는 라이브러리 버전이 서로 달라 직접 비교에서 제외했습니다.")
    for index, row in enumerate(rows):
        if index and row["comparable"] and rows[index - 1]["comparable"]:
            row["delta_previous_hartree"] = row["energy_hartree"] - rows[index - 1]["energy_hartree"]
            row["delta_previous_per_atom_hartree"] = row["energy_per_atom_hartree"] - rows[index - 1]["energy_per_atom_hartree"]

    total = len(expected_specs)
    completed = sum(row["status"] in TERMINAL for row in rows)
    successful = sum(row["status"] == "succeeded" for row in rows)
    valid = sum(row["comparable"] for row in rows)
    complete = completed == total
    comparable = valid == total and total > 0
    if complete:
        status = ("completed" if comparable else "cancelled" if record.get("cancel_requested")
                  else "completed_with_issues")
    else:
        status = "running" if completed or any(row["status"] == "running" for row in rows) else "queued"
    trend, total_change = None, None
    if comparable:
        deltas = [row["delta_previous_hartree"] for row in rows[1:]]
        trend = ("unchanged" if all(d == 0 for d in deltas)
                 else "decreased" if all(d <= 0 for d in deltas)
                 else "increased" if all(d >= 0 for d in deltas) else "mixed")
        total_change = rows[-1]["energy_hartree"] - rows[0]["energy_hartree"]

    messages = []
    if comparable:
        trend_text = {"decreased": "순서대로 낮아졌습니다", "increased": "순서대로 높아졌습니다",
                      "mixed": "증가와 감소가 함께 나타났습니다", "unchanged": "동일하게 나타났습니다"}[trend]
        messages.append({"kind": "observation", "text": f"이번 계산에서 {definition['variable_label']}을 바꿨을 때 전체 에너지는 {trend_text}."})
        if total > 1:
            messages.append({"kind": "observation", "text": (
                f"{labels[-2]} → {labels[-1]}의 ΔE는 {rows[-1]['delta_previous_hartree']:+.9f} Eh입니다. "
                "이는 두 계산의 차이이며 정확한 값에 대한 오차가 아닙니다.")})
        if definition["domain"] == "crystal" and total > 1:
            messages.append({"kind": "observation", "text": (
                f"같은 {len(expected_specs[0]['atoms'])}원자 셀에서 마지막 원자당 ΔE는 "
                f"{rows[-1]['delta_previous_per_atom_hartree']:+.9f} Eh/atom입니다. 전체 에너지 열은 단위격자 기준입니다.")})
        if definition["variable"] == "bond_distance":
            lowest = min(rows, key=lambda row: row["energy_hartree"])
            messages.append({"kind": "observation", "text": f"계산한 표본 중 최저 에너지는 {lowest['label']}에서 나타났습니다. 표본 사이의 최소점이나 평형 결합 길이를 계산한 것은 아닙니다."})
        if record.get("prediction") in {"lower", "higher"}:
            matches = {"lower": "decreased", "higher": "increased"}[record["prediction"]] == trend
            messages.append({"kind": "observation", "text": (
                "이번 관찰은 선택한 예상 방향과 일치합니다." if matches
                else "이번 관찰은 선택한 예상 방향과 일치하지 않습니다. 실제 결과를 기준으로 가정을 다시 살펴보세요.")})
    else:
        messages.append({"kind": "observation", "text": f"{total}개 중 비교 가능한 최종 결과는 {valid}개입니다. 대기·미수렴·취소·조건 불일치 값은 비교에서 제외합니다."})
    messages.extend([
        {"kind": "limit", "text": "SCF 수렴은 선택한 표현과 방법 안에서 수치 풀이가 수렴했다는 뜻입니다. 이 비교만으로 기저·격자·k점 수렴이나 실험값의 정확도를 인증하지 않습니다."},
        {"kind": "limit", "text": definition["limit"]},
        {"kind": "limit", "text": "시간과 최대 메모리는 이 컴퓨터에서 한 번 측정한 프로세스 값입니다. 엔진 준비 시간과 라이브러리 메모리를 포함하며 하드웨어에 따라 달라집니다."},
        {"kind": "next", "text": definition["next"]},
    ])
    record.update(
        definition=definition, domain=definition["domain"], variable=definition["variable"],
        variable_label=definition["variable_label"], labels=labels, values=values,
        status=status, completed_jobs=completed, successful_jobs=successful, total_jobs=total,
        active_job_id=next((row["job_id"] for row in rows if row["status"] == "running"), None),
        rows=rows, jobs=deepcopy(jobs),
        analysis={"valid_count": valid, "total_count": total, "complete": complete, "comparable": comparable,
                  "energy_trend": trend, "last_step_delta_hartree": rows[-1]["delta_previous_hartree"] if rows else None,
                  "total_change_hartree": total_change, "messages": messages,
                  "interpretation_version": INTERPRETATION_VERSION},
    )
    return record
