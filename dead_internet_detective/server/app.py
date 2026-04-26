from __future__ import annotations

import os
import threading
import uuid
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dead_internet_detective.environment import DeadInternetEnvironment
from dead_internet_detective.models import Action


app = FastAPI()

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    return (_static_dir / "index.html").read_text()
_sessions: dict[str, DeadInternetEnvironment] = {}
_lock = threading.Lock()


def _safe_asdict(obj: Any) -> Any:
    """asdict() replacement that converts sets → lists for JSON safety."""
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _safe_asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _safe_asdict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_asdict(i) for i in obj]
    return obj


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int | None = None


class StepRequest(BaseModel):
    session_id: str
    action: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(body: ResetRequest) -> dict:
    env = DeadInternetEnvironment()
    obs = env.reset(task_id=body.task_id, seed=body.seed)
    session_id = str(uuid.uuid4())
    with _lock:
        _sessions[session_id] = env
    return {"session_id": session_id, "observation": _safe_asdict(obs)}


@app.post("/step")
def step(body: StepRequest) -> dict:
    with _lock:
        env = _sessions.get(body.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")

    action = Action(tool=body.action["tool"], params=body.action.get("params", {}))
    obs, reward, done, info = env.step(action)

    if done:
        with _lock:
            _sessions.pop(body.session_id, None)

    return {
        "observation": _safe_asdict(obs),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state/{session_id}")
def get_state(session_id: str) -> dict:
    with _lock:
        env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _safe_asdict(env.state())
