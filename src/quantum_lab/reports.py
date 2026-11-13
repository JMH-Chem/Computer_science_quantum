"""Portable reports and numeric tables from retained results, without recomputation."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import zipfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from .storage import TERMINAL, dumps, now

router = APIRouter(prefix="/jobs", tags=["reports"])
BUNDLE_LIMIT = 256 * 1024 * 1024


def numeric_table(result: dict, kind: str) -> tuple[list[str], list[list]]:
    """Column names carry physical units; missing observations stay empty."""
    if kind == "excitations":
        columns = ["state", "energy_ev", "oscillator_strength", "converged"]
        rows = [[state.get(key) for key in columns] for state in result.get("excitations", [])]
    elif kind == "eom":
        columns = ["state", "energy_ev", "converged", "residual_norm"]
        rows = [[state.get(key) for key in columns] for state in (result.get("eom") or {}).get("states", [])]
    elif kind == "bands":
        data = result.get("bands") or {}
        columns = ["path_index", "distance_angstrom_inverse", "band_index", "energy_ev_relative_to_sampled_vbm", "break_before", "reference_energy_hartree"]
        rows = [[i, distance, j, energy, i in data.get("break_indices", []), data.get("reference_energy_hartree")]
                for i, (distance, energies) in enumerate(zip(data.get("distance", []), data.get("energies_ev", []), strict=True))
                for j, energy in enumerate(energies)]
    elif kind == "dos":
        data = result.get("sampled_dos") or {}
        columns = ["energy_ev_relative_to_scf_vbm", "density_states_per_ev_cell", "occupied_density_electrons_per_ev_cell", "reference_energy_hartree"]
        rows = [[e, d, occupied, data.get("reference_energy_hartree")]
                for e, d, occupied in zip(data.get("energy_ev", []), data.get("density_states_per_ev", []),
                                          data.get("occupied_density_states_per_ev", []), strict=True)]
    elif kind == "frequencies":
        columns = ["mode_index", "frequency_cm1", "reduced_mass_amu", "ir_intensity_km_mol", "imaginary"]
        rows = [[m.get("index"), m.get("frequency_cm1"), m.get("reduced_mass_amu"), m.get("ir_intensity_km_mol"), m.get("imaginary")]
                for m in (result.get("vibrations") or {}).get("modes", [])]
    elif kind == "charges":
        data = result.get("properties") or {}
        columns = ["atom_index", "element", "mulliken_charge_e", "lowdin_charge_e", "density_reference"]
        rows = [[i, atom["symbol"], m, low, data.get("reference_model")]
                for i, (atom, m, low) in enumerate(zip(result.get("atoms", []), data.get("mulliken_charges", []),
                                                       data.get("lowdin_charges", []), strict=True))] if data else []
    elif kind == "optimization":
        columns = ["step", "energy_hartree", "scf_converged"]
        rows = [[step.get(key) for key in columns] for step in result.get("optimization", [])]
    elif kind == "correlation":
        data = result.get("correlation") or {}
        columns = ["method", "reference_energy_hartree", "total_energy_hartree", "correlation_energy_hartree", "converged"]
        rows = [[data.get(key) for key in columns]] if data else []
    else:
        raise ValueError("지원 데이터: excitations, eom, bands, dos, frequencies, charges, optimization, correlation")
    if not rows:
        raise ValueError("이 작업에는 요청한 계산 데이터가 없습니다.")
    return columns, rows


def _csv(columns, rows) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(columns)
    writer.writerows(rows)
    return out.getvalue()


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _table(columns, rows) -> str:
    return "<table><thead><tr>" + "".join(f"<th>{_escape(c)}</th>" for c in columns) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{_escape('—' if v is None else v)}</td>" for v in row) + "</tr>" for row in rows
    ) + "</tbody></table>"


def _scf_plot(events) -> str:
    values = [event["data"].get("energy_hartree") for event in events if event["type"] == "scf_iteration"]
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    if not values:
        return "<p>저장된 SCF 반복 값이 없습니다.</p>"
    low, high = min(values), max(values)
    span = max(high - low, 1e-12)
    points = " ".join(f"{50 + 700*i/max(len(values)-1,1):.3f},{180-140*(v-low)/span:.3f}" for i, v in enumerate(values))
    return (f'<svg role="img" aria-label="저장된 SCF 에너지 이력" viewBox="0 0 800 215">'
            f'<path d="M50 30 V180 H760" stroke="#899" fill="none"/>'
            f'<polyline points="{points}" fill="none" stroke="#087e75" stroke-width="2"/>'
            f'<text x="50" y="22">{high:.9f} Eh</text><text x="50" y="205">{low:.9f} Eh · 기록 순서 (시간 아님)</text></svg>')


def report_html(job: dict, events: list[dict]) -> str:
    result, spec = job.get("result") or {}, job["spec"]
    title = _escape(job["name"])
    sections = [f"<h1>{title}</h1>", f"<p>계산 {job['id']} · 생성 {_escape(job['created_at'])} · 상태 <strong>{_escape(job['status'])}</strong></p>"]
    if job["status"] != "succeeded" or not result.get("converged"):
        sections.append("<p class=warning>이 작업은 최종 수렴한 성공 결과가 아닙니다. 중간 데이터와 오류를 성공한 물성값으로 해석하지 마세요.</p>")
    sections.append(_table(["항목", "값"], [
        ["대상 / 작업", f"{spec['domain']} / {spec['task']}"], ["방법", spec["method"]],
        ["기저", spec["basis"]], ["함수형 (DFT)", spec["functional"] if spec["method"] == "DFT" else "—"],
        ["전체 에너지 / Eh", result.get("energy_hartree")], ["엔진", result.get("engine")],
        ["밀도 기준", result.get("density_reference", (result.get("properties") or {}).get("reference_model"))],
    ]))
    sections.extend(["<h2>SCF 수렴 이력</h2>", _scf_plot(events),
                     "<p>반복 번호는 시간이 아닙니다. SCF 수렴과 기저·격자·k점 수렴은 별도로 판단합니다.</p>"])
    for kind, label in [("correlation", "전자 상관"), ("charges", "원자 전하 · 정의와 기저에 의존"),
                        ("excitations", "수직 여기 상태"), ("eom", "EOM-EE-CCSD singlet 여기 에너지 · 전이 세기 미계산"),
                        ("frequencies", "조화 진동수 · 음수는 허수 모드"),
                        ("optimization", "구조 최적화 단계 · 실제 시간 궤적 아님")]:
        try:
            columns, rows = numeric_table(result, kind)
            sections += [f"<h2>{label}</h2>", _table(columns, rows)]
        except ValueError:
            pass
    atoms = result.get("atoms") or spec["atoms"]
    sections += ["<h2>최종 보관 구조 / Å</h2>", _table(["원소", "x", "y", "z"], [[a["symbol"], *a["position"]] for a in atoms])]
    for name in ("vibrations", "sampled_band_edges", "crystal_properties", "provenance", "run_environment"):
        if result.get(name):
            value = result[name]
            if name == "vibrations":
                value = {k: v for k, v in value.items() if k not in {"modes", "frequencies_cm1"}}
            sections += [f"<h2>{name}</h2>", f"<pre>{_escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"]
    warnings = list(result.get("warnings") or [])
    if job.get("error"):
        warnings.append(job["error"])
    sections += ["<h2>진단과 해석 범위</h2>", "<ul>" + "".join(f"<li>{_escape(w)}</li>" for w in warnings) + "</ul>",
                 "<p>오비탈 에너지·전체 에너지·광학 여기 에너지는 서로 다른 물리량입니다. 계산된 조건의 결과이며 실험값과의 일치를 보증하지 않습니다.</p>",
                 "<h2>전체 입력</h2>", f"<pre>{_escape(json.dumps(spec, ensure_ascii=False, indent=2))}</pre>"]
    return ("<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{title} · Quantum Lab 보고서</title><style>"
            "body{font:15px system-ui,sans-serif;color:#152b35;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.65}"
            "h1,h2{color:#086d66}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px}"
            "td,th{border-bottom:1px solid #cbd7dc;text-align:left;padding:8px;overflow-wrap:anywhere}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f6f6;padding:16px;font-size:12px}"
            ".warning{background:#fff0d5;padding:16px}svg{max-width:800px;width:100%}"
            "@media print{body{margin:0;max-width:none}tr,svg{break-inside:avoid}h2{break-after:avoid}}"
            "</style><body>" + "".join(sections) + "</body></html>")


@router.get("/{job_id}/data")
def data_export(job_id: str, request: Request, kind: str):
    job = request.app.state.store.get(job_id)
    try:
        columns, rows = numeric_table(job.get("result") or {}, kind)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(_csv(columns, rows), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{job_id}-{kind}.csv"',
    })


@router.get("/{job_id}/report", response_class=HTMLResponse)
def report(job_id: str, request: Request):
    store = request.app.state.store
    return HTMLResponse(report_html(store.get(job_id), store.events(job_id)), headers={
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors 'self'",
        "X-Content-Type-Options": "nosniff",
    })


@router.get("/{job_id}/bundle")
def bundle(job_id: str, request: Request):
    store = request.app.state.store
    job = store.get(job_id)
    if job["status"] not in TERMINAL:
        raise HTTPException(409, "실행 중에는 파일이 바뀔 수 있습니다. 작업 종료 후 재현 묶음을 다운로드하세요.")
    directory = store.job_dir(job_id)
    paths = [p for p in sorted(directory.iterdir()) if p.is_file() and not p.is_symlink() and not p.name.endswith(".tmp")]
    if sum(p.stat().st_size for p in paths) > BUNDLE_LIMIT:
        raise HTTPException(413, "묶음은 256 MiB 이하에서 제공합니다. 큰 격자·체크포인트는 개별 파일로 다운로드하세요.")
    events = store.events(job_id)
    output, manifest = io.BytesIO(), {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        def add(name, content):
            raw = content.encode("utf-8") if isinstance(content, str) else content
            archive.writestr(name, raw)
            manifest[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

        add("job.json", dumps({**job, "events": events}))
        add("input.json", dumps(job["spec"]))
        add("report.html", report_html(job, events))
        add("README.txt", "Quantum Lab 계산 보관 묶음\n\ninput.json: 원래 계산 입력\njob.json: 상태, 결과, 엔진 버전, 이벤트\nartifacts/: 보관된 기저 정의·체크포인트·로그·배열\nmanifest.json: 파일 크기와 SHA-256\n\n동일한 Quantum Lab/엔진 버전과 의존성으로 재실행하세요. 현재 설치한 버전이 과거 계산의 버전과 같다고 가정하지 않습니다.\n새 계산: quantum-lab run input.json --data-dir ./recalculated\nvolume 작업은 원본 계산과 체크포인트가 필요하며 이 명령으로 단독 재실행하지 않습니다.\n")
        for path in paths:
            add("artifacts/" + path.name, path.read_bytes())
        archive.writestr("manifest.json", dumps({"format_version": 1, "job_id": job_id, "exported_at": now(), "files": manifest}))
    return Response(output.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="quantum-lab-{job_id}.zip"',
    })
