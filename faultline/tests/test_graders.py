import pytest
from faultline.env import FaultLineEnv
from faultline.graders.grader_easy import GraderEasy
from faultline.graders.grader_medium import GraderMedium
from faultline.graders.grader_hard import GraderHard
from faultline.models import (
    AcknowledgeAlertAction, QueryLogsAction, CheckMetricsAction,
    ResolveAction, RollbackAction, ScaleServiceAction,
)

class TestGraderEasy:
    def _run_perfect_episode(self):
        env = FaultLineEnv("single_service_latency", seed=42)
        env.reset()
        env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        env.step(QueryLogsAction(service="elasticsearch", time_range="last_15m"))
        env.step(CheckMetricsAction(service="elasticsearch", metric_name="memory", window_minutes=15))
        env.step(ResolveAction(
            root_cause_service="elasticsearch",
            postmortem_text="Elasticsearch ran out of heap memory due to GC pressure causing OOM events. Search-service p99 latency exceeded 2000ms. Remediation: increase heap size and tune GC settings."
        ))
        return env

    def test_perfect_episode_scores_above_0_8(self):
        env = self._run_perfect_episode()
        result = GraderEasy().grade(env._task)
        assert result.score >= 0.80

    def test_correct_root_cause_required_for_pass(self):
        env = FaultLineEnv("single_service_latency", seed=42)
        env.reset()
        env.step(ResolveAction(root_cause_service="search-service", postmortem_text="search was slow"))
        result = GraderEasy().grade(env._task)
        assert result.passed is False

    def test_score_in_valid_range(self):
        env = self._run_perfect_episode()
        result = GraderEasy().grade(env._task)
        assert 0.0 <= result.score <= 1.0

    def test_score_is_deterministic(self):
        env1 = self._run_perfect_episode()
        env2 = self._run_perfect_episode()
        r1 = GraderEasy().grade(env1._task)
        r2 = GraderEasy().grade(env2._task)
        assert r1.score == r2.score

class TestGraderMedium:
    def _run_medium_episode(self):
        env = FaultLineEnv("cascading_failure", seed=99)
        env.reset()
        env.step(AcknowledgeAlertAction(alert_id="alert-002"))
        env.step(AcknowledgeAlertAction(alert_id="alert-004"))
        env.step(QueryLogsAction(service="payment-db", time_range="last_15m"))
        env.step(QueryLogsAction(service="payment-service", time_range="last_15m"))
        env.step(RollbackAction(service="payment-service", target_version="v1.4.1"))
        return env

    def test_correct_rollback_scores_above_0_5(self):
        env = self._run_medium_episode()
        result = GraderMedium().grade(env._task)
        assert result.score >= 0.50

    def test_score_in_valid_range(self):
        env = self._run_medium_episode()
        result = GraderMedium().grade(env._task)
        assert 0.0 <= result.score <= 1.0

    def test_wrong_rollback_fails(self):
        env = FaultLineEnv("cascading_failure", seed=99)
        env.reset()
        env.step(RollbackAction(service="order-service", target_version="v2.0.0"))
        result = GraderMedium().grade(env._task)
        assert result.passed is False

class TestGraderHard:
    def _run_hard_episode(self):
        env = FaultLineEnv("multi_region_incident", seed=777)
        env.reset()
        env.step(AcknowledgeAlertAction(alert_id="alert-006"))
        env.step(AcknowledgeAlertAction(alert_id="alert-007"))
        env.step(QueryLogsAction(service="fraud-detector", time_range="last_15m"))
        env.step(QueryLogsAction(service="model-serving", time_range="last_15m"))
        env.step(CheckMetricsAction(service="model-serving", metric_name="cpu", window_minutes=15))
        env.step(ScaleServiceAction(service="model-serving", replicas=4))
        return env

    def test_correct_scale_scores_above_0_4(self):
        env = self._run_hard_episode()
        result = GraderHard().grade(env._task)
        assert result.score >= 0.40

    def test_score_in_valid_range(self):
        env = self._run_hard_episode()
        result = GraderHard().grade(env._task)
        assert 0.0 <= result.score <= 1.0

    def test_red_herring_investigation_penalised(self):
        env = FaultLineEnv("multi_region_incident", seed=777)
        env.reset()
        env.step(QueryLogsAction(service="elasticsearch", time_range="last_15m"))
        env.step(QueryLogsAction(service="elasticsearch", time_range="last_15m"))
        task = env._task
        assert task.reward_breakdown.get("red_herring_elasticsearch", 0.0) < 0
