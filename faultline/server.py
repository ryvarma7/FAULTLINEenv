from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

from faultline.env import FaultLineEnv
from faultline.models import FaultLineAction

app = FastAPI()
env = FaultLineEnv()


# ---------------------------
# REQUEST MODELS
# ---------------------------

class ResetRequest(BaseModel):
    task_id: str
    seed: int = 42
    config: Optional[dict] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ---------------------------
# ROUTES
# ---------------------------

@app.post("/reset")
def reset_env(req: ResetRequest):
    obs = env.reset(task_id=req.task_id, seed=req.seed, config=req.config)
    return {"observation": obs}


@app.post("/step")
def step_env(req: StepRequest):
    try:
        action = FaultLineAction(**req.action)
    except Exception as e:
        return {
            "observation": {},
            "reward": 0.0,
            "done": False,
            "info": {"error": f"INVALID_ACTION: {str(e)}"}
        }

    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def get_state():
    return env.state()


@app.get("/")
def root():
    return {"message": "FaultLine v2 server is running"}