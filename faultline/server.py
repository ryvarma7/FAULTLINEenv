"""
FaultLine mini-server (faultline/server.py)
===========================================
FastAPI server used by the faultline package directly.

HARDENING CHANGELOG:
- CENTRALIZED parsing: all action parsing now goes through
  faultline.utils.action_parser.parse_action() — single source of truth.
  DO NOT use FaultLineAction(**data) anywhere.
- STEP CONTRACT: every env.step() result is validated with
  faultline.utils.validators.validate_step_output() before return.
- STRUCTURED ERRORS: no silent failures; every code path returns a fully
  formed {observation, reward, done} dict with error context.
- DEBUG MODE: DEBUG=True emits [DEBUG] lines at every stage.
- EXCEPTION HANDLING: bare `except:` is forbidden; all handlers use
  `except Exception as e:` and always return structured error dicts.
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from faultline.env import FaultLineEnv
from faultline.models import FaultLineObservation
from faultline.utils.action_parser import parse_action
from faultline.utils.validators import validate_step_output
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Global debug flag — set to False in production to reduce log volume
# ---------------------------------------------------------------------------
DEBUG: bool = os.getenv("FAULTLINE_DEBUG", "True").lower() == "true"

app = FastAPI(
    title="FaultLine (mini)",
    description="FaultLine SRE environment — mini server",
    version="2.0.0",
)

env = FaultLineEnv()


# ---------------------------
# REQUEST MODELS
# ---------------------------

class ResetRequest(BaseModel):
    task_id: str = "single_service_latency"
    seed: int = 42
    config: Optional[dict] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ---------------------------
# HELPER: safe observation serializer
# ---------------------------

def _safe_obs(obs) -> Dict[str, Any]:
    """Serialize an observation safely — handles both Pydantic models and plain dicts."""
    if obs is None:
        return {}
    if isinstance(obs, FaultLineObservation):
        return obs.model_dump()
    if isinstance(obs, dict):
        return obs
    # Fallback: try model_dump, else str
    try:
        return obs.model_dump()
    except Exception:
        return {"raw": str(obs)}


# ---------------------------
# ROUTES
# ---------------------------

@app.post("/reset")
def reset_env(req: ResetRequest):
    """Reset the environment and return the initial observation."""
    if DEBUG:
        print(f"[DEBUG] /reset called: task_id={req.task_id} seed={req.seed}", flush=True)
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed, config=req.config)
        obs_dict = _safe_obs(obs)
        if DEBUG:
            print(f"[DEBUG] Reset complete. Alerts: {len(obs_dict.get('alerts', []))}", flush=True)
        return {"observation": obs_dict}
    except Exception as e:
        print(f"[ERROR] /reset failed: {e}", flush=True)
        traceback.print_exc()
        return {
            "observation": {},
            "error": str(e),
            "error_type": type(e).__name__,
        }


@app.post("/step")
def step_env(req: StepRequest):
    """
    Execute one action in the environment.

    Action parsing is delegated to parse_action() — the single source of
    truth that uses TypeAdapter(FaultLineAction).validate_python(). This is
    the ONLY correct way to instantiate Pydantic v2 discriminated unions.
    DO NOT use FaultLineAction(**data).
    """
    if DEBUG:
        print(f"[DEBUG] /step called. Incoming action: {req.action}", flush=True)

    # ----------------------------------------------------------------
    # STEP 1: Parse action via centralized parser — NEVER raises
    # ----------------------------------------------------------------
    action, error = parse_action(req.action)

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
    # STEP 2: Execute step — hardened with try/except + contract validation
    # ----------------------------------------------------------------
    if DEBUG:
        print(f"[DEBUG] Executing step with action type={action.type}", flush=True)

    try:
        result = env.step(action)  # returns dict, NOT tuple

        # Enforce step() output contract — raises descriptively if broken
        validate_step_output(result)

        if DEBUG:
            print(f"[DEBUG] Step result keys: {list(result.keys())}", flush=True)
            print(f"[DEBUG] Step result reward: {result.get('reward')}", flush=True)
            print(f"[DEBUG] Step result done: {result.get('done')}", flush=True)

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
    # STEP 3: Serialize observation and return
    # ----------------------------------------------------------------
    obs_raw = result.get("observation")
    result["observation"] = _safe_obs(obs_raw)

    reward = result.get("reward", 0.0)
    print(f"[REWARD] {reward}", flush=True)

    return result


@app.get("/state")
def get_state():
    """Return the current observation without advancing the episode."""
    return env.state()


@app.get("/")
def root():
    return {"message": "FaultLine v2 server is running", "version": "2.0.0"}