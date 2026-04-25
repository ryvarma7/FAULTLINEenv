"""
FaultLine main server (server/app.py)
======================================
Primary FastAPI application served via uvicorn.

HARDENING CHANGELOG:
- CENTRALIZED parsing: /step uses faultline.utils.action_parser.parse_action() — single
  source of truth. No more inline TypeAdapter or FaultLineAction(**data) anywhere.
- STEP CONTRACT: every env.step() result is validated with validate_step_output().
- STRUCTURED ERRORS: every code path returns {observation, reward, done} even on failure.
- DEBUG MODE: DEBUG=True emits [DEBUG] lines at every stage.
- BARE EXCEPT FORBIDDEN: all handlers use `except Exception as e:` with structured return.
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from faultline.env import FaultLineEnv, TASK_REGISTRY
from faultline.models import FaultLineObservation
from faultline.utils.action_parser import parse_action
from faultline.utils.validators import validate_step_output
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Global debug flag — set False in production to reduce log volume
# ---------------------------------------------------------------------------
DEBUG: bool = os.getenv("FAULTLINE_DEBUG", "True").lower() == "true"

app = FastAPI(
    title="FaultLine",
    description="SRE Incident Response Agent Environment — OpenEnv Hackathon",
    version="2.0.0",
)

# In-memory environment store (keyed by session_id, defaults to "default")
_envs: Dict[str, FaultLineEnv] = {}


# ---------------------------
# HELPERS
# ---------------------------

def _get_env(session_id: str = "default") -> FaultLineEnv:
    if session_id not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"No environment found for session '{session_id}'. Call /reset first.",
        )
    return _envs[session_id]


def _safe_obs(obs) -> Dict[str, Any]:
    """Safely serialize an observation — handles Pydantic models AND plain dicts."""
    if obs is None:
        return {}
    if isinstance(obs, FaultLineObservation):
        return obs.model_dump()
    if isinstance(obs, dict):
        return obs
    # Fallback for any other model type
    try:
        return obs.model_dump()
    except Exception:
        return {"raw": str(obs)}


def _safe_step_dict(result: dict, env: FaultLineEnv) -> dict:
    """
    Normalize step result dict:
    - Serialize observation safely
    - Guarantee reward is float
    - Guarantee done is bool
    """
    obs_raw = result.get("observation")
    result["observation"] = _safe_obs(obs_raw)

    reward = result.get("reward", 0.0)
    if not isinstance(reward, float):
        reward = float(reward)
    result["reward"] = reward

    result["done"] = bool(result.get("done", False))
    result.setdefault("info", {})

    return result


# ---------------------------
# REQUEST SCHEMAS
# ---------------------------

class ResetRequest(BaseModel):
    task_id: str = "single_service_latency"
    seed: int = 42
    session_id: str = "default"
    config: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: str = "default"


class StateRequest(BaseModel):
    session_id: str = "default"


class GradeRequest(BaseModel):
    session_id: str = "default"


# ---------------------------
# ENDPOINTS
# ---------------------------

@app.get("/")
async def health():
    return {
        "status": "ok",
        "name": "faultline",
        "version": "2.0.0",
        "tasks": list(TASK_REGISTRY.keys()),
    }


@app.post("/reset")
async def reset(request: ResetRequest = None):
    """Reset the environment and return the initial observation."""
    if request is None:
        request = ResetRequest()

    print(f"[DEBUG] /reset called: task_id={request.task_id} seed={request.seed} session={request.session_id}", flush=True)

    task_id = request.task_id
    if task_id not in TASK_REGISTRY and not request.config:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY)}",
        )

    try:
        env = FaultLineEnv(task_id=task_id, seed=request.seed)

        config_obj = None
        if request.config:
            from faultline.models import IncidentConfig
            config_obj = IncidentConfig(**request.config)

        obs = env.reset(task_id=task_id, seed=request.seed, config=config_obj)
        _envs[request.session_id] = env

        obs_dict = _safe_obs(obs)
        print(f"[DEBUG] Reset complete. Alerts: {len(obs_dict.get('alerts', []))}", flush=True)
        return obs_dict

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] /reset failed: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


@app.post("/step")
async def step(request: StepRequest):
    """
    Execute one action and return observation, reward, done, info.

    Action parsing is delegated to parse_action() — the centralized utility
    that uses TypeAdapter(FaultLineAction).validate_python(). This is the ONLY
    correct way to instantiate Pydantic v2 discriminated unions.
    DO NOT use FaultLineAction(**data) or action_map[type](**data) anywhere.
    """
    env = _get_env(request.session_id)

    if DEBUG:
        print(f"[DEBUG] /step incoming action: {request.action}", flush=True)

    # ----------------------------------------------------------------
    # STEP 1: Parse action via centralized parser — NEVER raises
    # ----------------------------------------------------------------
    action, error = parse_action(request.action)

    if error:
        print(f"[ERROR] Action parsing failed: {error['message']}", flush=True)
        current_obs = _safe_obs(getattr(env, "_current_observation", None))
        return {
            "observation": current_obs,
            "reward": 0.0,
            "done": False,
            "info": {},
            **error,
        }

    if DEBUG:
        print(f"[DEBUG] Parsed action: type={action.type}", flush=True)

    # ----------------------------------------------------------------
    # STEP 2: Execute step + enforce contract
    # ----------------------------------------------------------------
    if DEBUG:
        print(f"[DEBUG] Executing step with action type={action.type}", flush=True)

    try:
        result = env.step(action)

        # Enforce step() output contract — raises descriptively if broken
        validate_step_output(result)

        if DEBUG:
            print(f"[DEBUG] Step raw result: reward={result.get('reward')} done={result.get('done')}", flush=True)

    except Exception as e:
        print(f"[ERROR] env.step() raised or contract violated: {e}", flush=True)
        traceback.print_exc()
        current_obs = _safe_obs(getattr(env, "_current_observation", None))
        return {
            "observation": current_obs,
            "reward": 0.0,
            "done": False,
            "info": {},
            "error": str(e),
            "error_type": type(e).__name__,
        }

    # ----------------------------------------------------------------
    # STEP 3: Normalize and return
    # ----------------------------------------------------------------
    result = _safe_step_dict(result, env)
    reward = result["reward"]
    print(f"[REWARD] {reward}", flush=True)
    if DEBUG:
        print(f"[DEBUG] Step complete: done={result['done']}", flush=True)

    return result


@app.post("/state")
async def state(request: StateRequest = None):
    """Return the current observation without advancing the episode."""
    if request is None:
        request = StateRequest()
    env = _get_env(request.session_id)
    return env.state()


@app.post("/grade")
async def grade(request: GradeRequest = None):
    """Grade the completed episode and return score breakdown."""
    if request is None:
        request = GradeRequest()
    env = _get_env(request.session_id)
    try:
        return env.grade()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with their IDs and difficulty."""
    return {
        "tasks": [
            {
                "id": "single_service_latency",
                "difficulty": "easy",
                "description": "Identify root cause of elasticsearch OOM causing search latency",
            },
            {
                "id": "cascading_failure",
                "difficulty": "medium",
                "description": "Trace cascading DB connection exhaustion to payment-service deployment",
            },
            {
                "id": "multi_region_incident",
                "difficulty": "hard",
                "description": "Navigate multi-service incident with red herring alerts to find model-serving CPU quota exhaustion",
            },
        ]
    }


def main():
    host = os.getenv("FAULTLINE_HOST", "0.0.0.0")
    port = int(os.getenv("FAULTLINE_PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
