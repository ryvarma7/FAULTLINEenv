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
    
    # Send wrong action
    from faultline.models import RollbackAction
    wrong_action = RollbackAction(service="frontend", target_version="v1")
    result = env.step(wrong_action)
    assert result["done"] is False
    assert result["reward"] == 0.0
    
    # Send correct action
    from faultline.models import ScaleServiceAction
    correct_action = ScaleServiceAction(service="search-service", replicas=3)
    result = env.step(correct_action)
    assert result["done"] is True
    assert result["reward"] == 1.0
