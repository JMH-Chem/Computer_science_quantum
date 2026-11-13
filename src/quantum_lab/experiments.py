"""Persisted guided experiments run on the same sequential calculation queue."""
from __future__ import annotations

import csv
import io
import json
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .experiment_science import build_specs, summarize, templates
from .storage import ExperimentConflict, dumps, now

router = APIRouter(prefix="/experiments", tags=["experiments"])


class CreateExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template_id: Literal[
        "water_basis_v1", "water_grid_v1", "water_functional_v1", "hydrogen_dissociation_v1",
        "diamond_kmesh_v1", "diamond_mesh_v1", "water_correlation_v1",
    ] = "water_basis_v1"
    name: str = Field(default="물의 기저 비교", min_length=1, max_length=80)
    prediction: Literal["lower", "higher", "unsure"] = "unsure"
    request_id: UUID

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("실험 이름을 입력하세요.")
        return value


def _detail(request: Request, experiment_id: str) -> dict:
    experiment, jobs = request.app.state.store.experiment_bundle(experiment_id)
    return summarize(experiment, jobs)


@router.get("/templates")
def experiment_templates():
    return templates()


@router.get("")
def list_experiments(request: Request):
    return [_detail(request, experiment["id"]) for experiment in request.app.state.store.list_experiments()]


@router.post("", status_code=201)
def create_experiment(body: CreateExperiment, request: Request):
    payload = body.model_dump(mode="json")
    try:
        specs = build_specs(body.template_id, body.name)
        record = request.app.state.manager.submit_experiment(payload, specs)
    except ExperimentConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _detail(request, record["id"])


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str, request: Request):
    return _detail(request, experiment_id)


@router.post("/{experiment_id}/cancel")
def cancel_experiment(experiment_id: str, request: Request):
    request.app.state.manager.cancel_experiment(experiment_id)
    return _detail(request, experiment_id)


@router.get("/{experiment_id}/export")
def export_experiment(experiment_id: str, request: Request, format: str = "json"):
    detail = _detail(request, experiment_id)
    store = request.app.state.store
    if format == "json":
        # Include the actual basis definitions, citations and event records so
        # the downloaded comparison retains its numerical provenance offline.
        artifacts = {}
        unavailable_artifacts = {}
        for job in detail["jobs"]:
            files = {}
            unavailable = {}
            directory = store.job_dir(job["id"])
            for filename in ("basis.json", "basis-references.txt", "periodic_basis.json"):
                path = directory / filename
                try:
                    if path.is_file() and not path.is_symlink():
                        text = path.read_text(encoding="utf-8")
                        files[filename] = json.loads(text) if filename.endswith(".json") else text
                except json.JSONDecodeError:
                    unavailable[filename] = {
                        "reason": "invalid_json", "message": "Artifact is incomplete or invalid JSON; omitted.",
                    }
                except UnicodeError:
                    unavailable[filename] = {
                        "reason": "invalid_encoding", "message": "Artifact is not valid UTF-8; omitted.",
                    }
                except OSError:
                    unavailable[filename] = {
                        "reason": "unreadable", "message": "Artifact could not be read; omitted.",
                    }
            artifacts[job["id"]] = files
            if unavailable:
                unavailable_artifacts[job["id"]] = unavailable
        content = dumps({**detail, "export_format_version": 1, "exported_at": now(),
                         "job_artifacts": artifacts,
                         "unavailable_artifacts": unavailable_artifacts,
                         "events": {job["id"]: store.events(job["id"]) for job in detail["jobs"]}})
        media = "application/json"
    elif format == "csv":
        fields = ["ordinal", "job_id", "label", "value", "basis", "status", "comparable", "energy_hartree",
                  "delta_previous_hartree", "energy_per_atom_hartree", "delta_previous_per_atom_hartree",
                  "basis_functions", "iteration_count", "elapsed_seconds", "peak_memory_mb",
                  "error", "comparison_note"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detail["rows"])
        content, media = output.getvalue(), "text/csv"
    else:
        raise HTTPException(422, "실험 내보내기는 json 또는 csv를 지원합니다.")
    return Response(content, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="experiment-{experiment_id}.{format}"',
    })
