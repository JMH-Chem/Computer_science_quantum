"""Loopback HTTP interface with durable, replayable computation events."""
from __future__ import annotations

import asyncio
import csv
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .jobs import JobManager
from .models import JobSpec, VolumeRequest
from .storage import TERMINAL, Store, dumps, json_value


def create_app(data_dir: str | Path | None = None, *, worker_module="quantum_lab.worker") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(data_dir or os.environ.get("QUANTUM_LAB_DATA", ".quantum-lab"))
        app.state.manager = JobManager(app.state.store, worker_module=worker_module)
        app.state.manager.start()
        try:
            yield
        finally:
            app.state.manager.stop()

    app = FastAPI(title="Quantum Lab", version=__version__, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            try:
                allowed = urlsplit(origin).hostname in {"localhost", "127.0.0.1", "::1"}
            except ValueError:
                allowed = False
            if not allowed:
                return JSONResponse({"detail": "Only local browser origins can modify this local service."}, 403)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing_record(request, exc):
        return JSONResponse({"detail": "Job not found."}, status_code=404)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # A rejected NaN/Infinity remains in Pydantic's diagnostic input. Serialize
        # it as null, and exception contexts as text, rather than turning a 422 into 500.
        detail = json_value(jsonable_encoder(exc.errors(), custom_encoder={Exception: str}))
        return JSONResponse({"detail": detail}, status_code=422)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "queue": "single-worker", "storage": "sqlite"}

    @app.get("/api/jobs")
    def list_jobs():
        return app.state.store.list()

    @app.post("/api/jobs", status_code=201)
    def submit(spec: JobSpec):
        if spec.task == "volume":
            raise HTTPException(422, "Use the parent job's /volumes endpoint for volume generation.")
        return app.state.manager.submit(spec.model_dump(mode="json"))

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return app.state.store.get(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return app.state.manager.cancel(job_id)

    @app.get("/api/jobs/{job_id}/history")
    def history(job_id: str, after: int = Query(0, ge=0)):
        return app.state.store.events(job_id, after)

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, request: Request, after: int = Query(0, ge=0)):
        app.state.store.get(job_id)
        try:
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            raise HTTPException(400, "Last-Event-ID must be an integer sequence number.")

        async def generate():
            nonlocal cursor
            idle = 0
            while not await request.is_disconnected():
                batch = app.state.store.events(job_id, cursor)
                for event in batch:
                    cursor = event["seq"]
                    yield f"id: {cursor}\ndata: {dumps(event)}\n\n"
                if app.state.store.get(job_id)["status"] in TERMINAL:
                    # A transition can commit between the preceding two reads.
                    tail = app.state.store.events(job_id, cursor)
                    for event in tail:
                        cursor = event["seq"]
                        yield f"id: {cursor}\ndata: {dumps(event)}\n\n"
                    break
                idle = 0 if batch else idle + 1
                if idle >= 40:
                    yield ": heartbeat\n\n"
                    idle = 0
                await asyncio.sleep(0.25)

        return StreamingResponse(generate(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive",
        })

    @app.get("/api/jobs/{job_id}/artifacts")
    def artifacts(job_id: str):
        app.state.store.get(job_id)
        directory = app.state.store.job_dir(job_id)
        return [{"name": p.name, "size": p.stat().st_size} for p in sorted(directory.iterdir())
                if p.is_file() and not p.is_symlink() and not p.name.endswith(".tmp")]

    @app.get("/api/jobs/{job_id}/artifacts/{filename}")
    def artifact(job_id: str, filename: str):
        app.state.store.get(job_id)
        directory = app.state.store.job_dir(job_id)
        path = directory / filename
        if path.parent != directory or path.is_symlink() or not path.is_file() or filename.endswith(".tmp"):
            raise HTTPException(404, "Artifact not found.")
        return FileResponse(path, filename=filename, media_type="application/octet-stream")

    @app.post("/api/jobs/{job_id}/volumes", status_code=201)
    def volume(job_id: str, request: VolumeRequest):
        parent = app.state.store.get(job_id)
        if parent["status"] != "succeeded" or parent["spec"]["task"] == "volume":
            raise HTTPException(409, "A converged calculation is required for a volume.")
        payload = request.model_dump(mode="json")
        if parent["spec"]["domain"] == "crystal" and payload["kind"] != "density":
            raise HTTPException(422, "Periodic volumes currently support electron density only.")
        spec = {**parent["spec"], "task": "volume", "parent_job_id": job_id, "volume": payload,
                "name": f"{parent['name'][:95]} · {payload['kind']}"}
        validated = JobSpec.model_validate(spec)
        return app.state.manager.submit_volume(validated.model_dump(mode="json"))

    @app.get("/api/jobs/{job_id}/export")
    def export(job_id: str, format: str = "json"):
        job = app.state.store.get(job_id)
        history = app.state.store.events(job_id)
        if format == "json":
            content = dumps({**job, "events": history})
            media = "application/json"
        elif format == "csv":
            out = io.StringIO()
            fields = ["seq", "type", "iteration", "optimization_step", "step", "energy_hartree",
                      "delta_energy", "gradient_norm", "density_change", "status"]
            writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for event in history:
                writer.writerow({"seq": event["seq"], "type": event["type"], **event["data"]})
            content, media = out.getvalue(), "text/csv"
        elif format in {"xyz", "cif"}:
            result = job["result"] or {}
            atoms = result.get("atoms") or job["spec"]["atoms"]
            if format == "xyz":
                lines = [str(len(atoms)), f"Quantum Lab {job_id}; coordinates in Angstrom; status={job['status']}"]
                lines.extend(f"{a['symbol']} " + " ".join(f"{x:.12f}" for x in a["position"]) for a in atoms)
                content, media = "\n".join(lines) + "\n", "chemical/x-xyz"
            else:
                lattice = result.get("lattice") or job["spec"].get("lattice")
                if lattice is None:
                    raise HTTPException(422, "CIF export requires a periodic lattice.")
                from ase import Atoms
                from ase.io import write
                out = io.BytesIO()
                structure = Atoms(symbols=[a["symbol"] for a in atoms], positions=[a["position"] for a in atoms],
                                  cell=lattice, pbc=True)
                write(out, structure, format="cif")
                content, media = out.getvalue(), "chemical/x-cif"
        else:
            raise HTTPException(422, "Supported formats: json, csv, xyz, cif.")
        return Response(content, media_type=media, headers={
            "Content-Disposition": f'attachment; filename="{job_id}.{format}"',
        })

    from .catalog import router as catalog_router
    from .structures import router as structures_router
    app.include_router(catalog_router, prefix="/api")
    app.include_router(structures_router, prefix="/api")

    frontend = Path(__file__).resolve().parent / "static"
    if not frontend.is_dir():
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend():
            return {"service": "Quantum Lab", "message": "Build frontend with npm run build in frontend/."}
    return app


app = create_app()
