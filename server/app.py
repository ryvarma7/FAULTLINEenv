from __future__ import annotations
import os
import uvicorn
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from faultline.env import FaultLineEnv, TASK_REGISTRY
from faultline.models import (
    FaultLineAction, FaultLineObservation, StepResult,
    QueryLogsAction, CheckMetricsAction, AcknowledgeAlertAction,
    RollbackAction, ScaleServiceAction, EscalateAction, ResolveAction,
)

app = FastAPI(
    title="FaultLine",
    description="SRE Incident Response Agent Environment — OpenEnv Hackathon",
    version="1.0.0",
)

# In-memory environment store (keyed by session_id, defaults to "default")
_envs: Dict[str, FaultLineEnv] = {}

def _get_env(session_id: str = "default") -> FaultLineEnv:
    if session_id not in _envs:
        raise HTTPException(status_code=400, detail=f"No environment found for session '{session_id}'. Call /reset first.")
    return _envs[session_id]

def _obs_to_dict(obs: FaultLineObservation) -> Dict[str, Any]:
    return obs.model_dump()

def _result_to_dict(result: StepResult) -> Dict[str, Any]:
    return {
        "observation": _obs_to_dict(result.observation),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }

# --- Request schemas ---

class ResetRequest(BaseModel):
    task_id: str = "single_service_latency"
    seed: int = 42
    session_id: str = "default"

class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: str = "default"

class StateRequest(BaseModel):
    session_id: str = "default"

class GradeRequest(BaseModel):
    session_id: str = "default"

# --- Endpoints ---

@app.get("/")
async def health():
    return {
        "status": "ok",
        "name": "faultline",
        "version": "1.0.0",
        "tasks": list(TASK_REGISTRY.keys()),
    }

@app.post("/reset")
async def reset(request: ResetRequest = None):
    """Reset the environment and return the initial observation."""
    if request is None:
        request = ResetRequest()
    task_id = request.task_id
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY)}")
    env = FaultLineEnv(task_id=task_id, seed=request.seed)
    result = env.reset()
    _envs[request.session_id] = env
    return _result_to_dict(result)

@app.post("/step")
async def step(request: StepRequest):
    """Execute one action and return the resulting observation, reward, done, info."""
    env = _get_env(request.session_id)
    try:
        action_data = request.action
        action_type = action_data.get("type")
        action_map = {
            "query_logs": QueryLogsAction,
            "check_metrics": CheckMetricsAction,
            "acknowledge_alert": AcknowledgeAlertAction,
            "rollback": RollbackAction,
            "scale_service": ScaleServiceAction,
            "escalate": EscalateAction,
            "resolve": ResolveAction,
        }
        if action_type not in action_map:
            raise HTTPException(status_code=400, detail=f"Unknown action type '{action_type}'. Valid: {list(action_map)}")
        action_obj = action_map[action_type](**action_data)
    except (KeyError, ValidationError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid action payload: {e}")
    result = env.step(action_obj)
    return _result_to_dict(result)

@app.post("/state")
async def state(request: StateRequest = None):
    """Return the current observation without advancing the episode."""
    if request is None:
        request = StateRequest()
    env = _get_env(request.session_id)
    obs = env.state()
    return _obs_to_dict(obs)

@app.post("/grade")
async def grade(request: GradeRequest = None):
    """Grade the completed episode and return score breakdown."""
    if request is None:
        request = GradeRequest()
    env = _get_env(request.session_id)
    return env.grade()

@app.get("/tasks")
async def list_tasks():
    """List all available tasks with their IDs and difficulty."""
    return {
        "tasks": [
            {"id": "single_service_latency", "difficulty": "easy",
             "description": "Identify root cause of elasticsearch OOM causing search latency"},
            {"id": "cascading_failure", "difficulty": "medium",
             "description": "Trace cascading DB connection exhaustion to payment-service deployment"},
            {"id": "multi_region_incident", "difficulty": "hard",
             "description": "Navigate multi-service incident with red herring alerts to find model-serving CPU quota exhaustion"},
        ]
    }


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
