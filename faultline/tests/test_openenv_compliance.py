import pytest
from faultline.env import FaultLineEnv
from faultline.models import FaultLineObservation, FaultLineAction, IncidentConfig

def test_reset_schema():
    env = FaultLineEnv(task_id="single_service_latency", seed=42)
    obs = env.reset()
    assert isinstance(obs, FaultLineObservation)
    
def test_step_schema():
    env = FaultLineEnv(task_id="single_service_latency", seed=42)
    env.reset()
    action = {"type": "query_logs", "service": "frontend", "time_range": "15m"}
    from faultline.models import QueryLogsAction
    act = QueryLogsAction(**action)
    
    result = env.step(act)
    assert isinstance(result, dict)
    assert "observation" in result
    assert "reward" in result
    assert "done" in result
    assert "info" in result
    
def test_state_schema():
    env = FaultLineEnv(task_id="single_service_latency", seed=42)
    env.reset()
    state_result = env.state()
    assert isinstance(state_result, dict)
    
def test_episode_termination_scenario():
    env = FaultLineEnv(task_id="single_service_latency", seed=42)
    config = IncidentConfig(
        failure_mode="latency",
        cascade_depth=1,
        red_herring_count=1,
        noise_level=0.1
    )
    env.reset(config=config)

    # A wrong terminal action (rollback when the scenario's correct action differs)
    # CORRECT behavior: wrong terminal action MUST end the episode (done=True)
    # with a negative penalty reward. Previous test had wrong expectations (done=False, reward=0.0)
    # which was masking the reward clamp bug (max(0.0,...) was clamping -0.20 to 0.0).
    from faultline.models import RollbackAction
    wrong_action = RollbackAction(service="frontend", target_version="v1")
    result = env.step(wrong_action)
    # Wrong terminal action ends the episode
    assert result["done"] is True, "Wrong terminal action must end the episode"
    # Penalty reward — must be strictly negative (clamp now allows down to -1.0)
    assert result["reward"] < 0.0, (
        f"Wrong terminal action must give negative reward, got {result['reward']}"
    )
    # Core contract keys present
    assert result["observation"] is not None
    assert "reward" in result
    assert "done" in result
    assert "info" in result
