import pytest
from faultline.env import FaultLineEnv
from faultline.models import (
    AcknowledgeAlertAction, QueryLogsAction, CheckMetricsAction,
    ResolveAction, RollbackAction, ScaleServiceAction, EscalateAction,
    EpisodeState,
)

class TestFaultLineEnvReset:
    def test_reset_easy_returns_observation(self):
        env = FaultLineEnv("single_service_latency", seed=42)
        result = env.reset()
        assert isinstance(result, __import__("faultline.models", fromlist=["FaultLineObservation"]).FaultLineObservation)
        assert len(result.alerts) == 1
        assert result.alerts[0].service == "search-service"

    def test_reset_medium_returns_three_alerts(self):
        env = FaultLineEnv("cascading_failure", seed=99)
        result = env.reset()
        assert len(result.alerts) == 3

    def test_reset_hard_returns_five_alerts(self):
        env = FaultLineEnv("multi_region_incident", seed=777)
        result = env.reset()
        assert len(result.alerts) == 5

    def test_reset_is_reproducible(self):
        env1 = FaultLineEnv("single_service_latency", seed=42)
        env2 = FaultLineEnv("single_service_latency", seed=42)
        r1 = env1.reset()
        r2 = env2.reset()
        assert r1.alerts[0].id == r2.alerts[0].id
        assert r1.alerts[0].description == r2.alerts[0].description

    def test_dependency_graph_present(self):
        env = FaultLineEnv("single_service_latency", seed=42)
        result = env.reset()
        assert "api-gateway" in result.dependency_graph
        assert "elasticsearch" in result.dependency_graph

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            FaultLineEnv("nonexistent_task", seed=42)

class TestFaultLineEnvStep:
    def setup_method(self):
        self.env = FaultLineEnv("single_service_latency", seed=42)
        self.env.reset()

    def test_acknowledge_alert_gives_reward(self):
        result = self.env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        assert result["reward"] > 0.0
        assert "alert-001" in result["observation"].acknowledged_alerts

    def test_query_logs_returns_log_entries(self):
        result = self.env.step(QueryLogsAction(service="elasticsearch", time_range="last_15m"))
        assert result["observation"].log_results is not None
        assert len(result["observation"].log_results) > 0

    def test_check_metrics_returns_series(self):
        result = self.env.step(CheckMetricsAction(service="elasticsearch", metric_name="memory", window_minutes=15))
        assert result["observation"].metric_results is not None
        assert result["observation"].metric_results.service == "elasticsearch"
        assert len(result["observation"].metric_results.points) == 15

    def test_resolve_correct_root_cause_ends_episode(self):
        self.env.step(ResolveAction(
            root_cause_service="elasticsearch",
            postmortem_text="Elasticsearch ran out of heap memory due to GC pressure. OOM events caused search-service timeouts.",
        ))
        result = self.env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        assert result["done"] is True

    def test_resolve_wrong_root_cause_gives_penalty(self):
        result = self.env.step(ResolveAction(
            root_cause_service="search-service",
            postmortem_text="search-service was slow",
        ))
        assert result["reward"] < 0 or result["done"] is True

    def test_elapsed_steps_increments(self):
        self.env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        obs = self.env.state()
        assert obs["elapsed_steps"] == 1

    def test_loop_penalty_applied_after_4_queries(self):
        for _ in range(4):
            self.env.step(QueryLogsAction(service="search-service", time_range="last_15m"))
        result = self.env.step(QueryLogsAction(service="search-service", time_range="last_15m"))
        # By the 4th+ query on same service/range, penalty should have been applied
        task = self.env._task
        assert task.reward_breakdown.get("loop_penalty", 0.0) < 0

    def test_state_returns_current_observation(self):
        self.env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        obs = self.env.state()
        assert obs["elapsed_steps"] == 1
        assert "alert-001" in obs["acknowledged_alerts"]

class TestEpisodeBoundaries:
    def test_max_steps_ends_episode(self):
        env = FaultLineEnv("single_service_latency", seed=42)
        env.reset()
        for _ in range(20):
            result = env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        assert result["done"] is True

    def test_escalate_ends_episode(self):
        env = FaultLineEnv("single_service_latency", seed=42)
        env.reset()
        result = env.step(EscalateAction(team="platform-sre", message="Cannot determine root cause"))
        assert result["done"] is True
