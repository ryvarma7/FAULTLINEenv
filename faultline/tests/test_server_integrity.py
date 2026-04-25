"""
tests/test_server_integrity.py
================================
Mandatory integrity tests for the FaultLine hardening pass.

Tests:
  1. Valid action → reward != always 0
  2. Invalid action → returns error dict, does NOT crash
  3. step() response always contains {observation, reward, done}
  4. reward changes across different actions
  5. TypeAdapter parses ALL 8 action types correctly

These tests exercise the DIRECT Python API (FaultLineEnv + TypeAdapter),
not the HTTP layer, so they run without a server process.
"""
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from faultline.env import FaultLineEnv
from faultline.models import (
    FaultLineAction,
    AcknowledgeAlertAction,
    QueryLogsAction,
    CheckMetricsAction,
    ResolveAction,
    RollbackAction,
    ScaleServiceAction,
    EscalateAction,
    QueryRunbookAction,
)

# Pre-built adapter — mirrors what server.py now uses
_ACTION_ADAPTER: TypeAdapter[FaultLineAction] = TypeAdapter(FaultLineAction)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh FaultLineEnv for each test."""
    e = FaultLineEnv("single_service_latency", seed=42)
    e.reset()
    return e


# ---------------------------------------------------------------------------
# TEST 1: Valid action → reward is not ALWAYS 0
# ---------------------------------------------------------------------------

class TestValidActionReward:
    def test_acknowledge_alert_reward_nonzero(self, env):
        """Acknowledging a P1/P2 alert MUST produce reward > 0."""
        result = env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        assert "reward" in result, "step() must return 'reward' key"
        assert result["reward"] > 0.0, (
            f"Expected reward > 0 for ack of P1/P2 alert, got {result['reward']}"
        )

    def test_query_root_cause_logs_reward_nonzero(self, env):
        """Querying logs for the root-cause service (elasticsearch) must give reward > 0."""
        result = env.step(QueryLogsAction(service="elasticsearch", time_range="last_15m"))
        assert result["reward"] > 0.0, (
            f"Expected reward > 0 for root-cause log query, got {result['reward']}"
        )

    def test_correct_resolve_reward_nonzero(self, env):
        """Correct resolve action must produce the largest reward."""
        result = env.step(ResolveAction(
            root_cause_service="elasticsearch",
            postmortem_text=(
                "Elasticsearch ran out of heap memory (OOM). "
                "GC pressure caused search-service latency spike. "
                "Remediation: increased heap and disabled aggressive caching."
            ),
        ))
        assert result["reward"] > 0.0, (
            f"Correct resolve should give reward > 0, got {result['reward']}"
        )
        assert result["done"] is True, "Correct resolve must end episode"


# ---------------------------------------------------------------------------
# TEST 2: Invalid action → returns error dict, does NOT crash or raise
# ---------------------------------------------------------------------------

class TestInvalidActionHandling:
    def test_unknown_type_returns_error_not_crash(self, env):
        """TypeAdapter should raise ValidationError for unknown type — caller handles it."""
        with pytest.raises(ValidationError):
            _ACTION_ADAPTER.validate_python({"type": "nuke_datacenter"})

    def test_missing_required_field_raises_validation_error(self):
        """Missing required field in action raises ValidationError cleanly."""
        with pytest.raises(ValidationError):
            _ACTION_ADAPTER.validate_python({"type": "query_logs"})  # missing service, time_range

    def test_env_unknown_action_type_returns_safe_dict(self, env):
        """
        If an action with unknown type somehow reaches env.step(), it should return
        a safe dict with reward=0.0 and done=False — not raise.
        We test via a mock-like approach: temporarily pass an object with bad type.
        """
        # Construct a valid action then monkey-patch its type to simulate bypass
        action = QueryLogsAction(service="api-gateway", time_range="last_15m")
        object.__setattr__(action, 'type', 'totally_invalid')

        result = env.step(action)  # Must NOT raise
        assert isinstance(result, dict), "step() must return a dict even for invalid actions"
        assert "reward" in result
        assert "done" in result
        assert "observation" in result
        assert result["reward"] == 0.0
        assert result["done"] is False


# ---------------------------------------------------------------------------
# TEST 3: step() response ALWAYS contains {observation, reward, done}
# ---------------------------------------------------------------------------

class TestStepResponseContract:
    REQUIRED_KEYS = {"observation", "reward", "done"}

    def _assert_contract(self, result):
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"step() response missing keys: {missing}"
        assert isinstance(result["reward"], float), f"reward must be float, got {type(result['reward'])}"
        assert isinstance(result["done"], bool), f"done must be bool, got {type(result['done'])}"
        assert result["observation"] is not None, "observation must not be None"

    def test_ack_action_contract(self, env):
        result = env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        self._assert_contract(result)

    def test_query_logs_contract(self, env):
        result = env.step(QueryLogsAction(service="elasticsearch", time_range="last_15m"))
        self._assert_contract(result)

    def test_check_metrics_contract(self, env):
        result = env.step(CheckMetricsAction(service="elasticsearch", metric_name="memory"))
        self._assert_contract(result)

    def test_resolve_contract(self, env):
        result = env.step(ResolveAction(root_cause_service="elasticsearch", postmortem_text="OOM"))
        self._assert_contract(result)

    def test_escalate_contract(self, env):
        result = env.step(EscalateAction(team="sre-oncall", message="Cannot determine root cause"))
        self._assert_contract(result)


# ---------------------------------------------------------------------------
# TEST 4: reward changes across actions (not always identical)
# ---------------------------------------------------------------------------

class TestRewardVariation:
    def test_rewards_not_all_zero_in_episode(self, env):
        """At least one step in a standard episode must produce non-zero reward."""
        actions = [
            AcknowledgeAlertAction(alert_id="alert-001"),
            QueryLogsAction(service="elasticsearch", time_range="last_15m"),
            CheckMetricsAction(service="elasticsearch", metric_name="memory"),
        ]
        rewards = [env.step(a)["reward"] for a in actions]
        assert any(r != 0.0 for r in rewards), (
            f"All rewards were 0.0 — reward pipeline is broken. rewards={rewards}"
        )

    def test_correct_vs_wrong_resolve_differ(self):
        """Correct resolve reward must differ from wrong resolve reward."""
        env_correct = FaultLineEnv("single_service_latency", seed=42)
        env_correct.reset()
        r_correct = env_correct.step(ResolveAction(
            root_cause_service="elasticsearch",
            postmortem_text="OOM in elasticsearch caused search latency",
        ))["reward"]

        env_wrong = FaultLineEnv("single_service_latency", seed=42)
        env_wrong.reset()
        r_wrong = env_wrong.step(ResolveAction(
            root_cause_service="api-gateway",
            postmortem_text="api-gateway was slow",
        ))["reward"]

        assert r_correct != r_wrong, (
            f"Correct and wrong resolve should give different rewards: "
            f"correct={r_correct} wrong={r_wrong}"
        )
        assert r_correct > r_wrong, "Correct resolve must outperform wrong resolve"

    def test_reward_range_valid(self, env):
        """All rewards must be in [-1.0, 1.0]."""
        actions = [
            AcknowledgeAlertAction(alert_id="alert-001"),
            QueryLogsAction(service="elasticsearch", time_range="last_15m"),
            CheckMetricsAction(service="elasticsearch", metric_name="memory"),
            CheckMetricsAction(service="search-service", metric_name="latency_p99"),
        ]
        for a in actions:
            result = env.step(a)
            r = result["reward"]
            assert -1.0 <= r <= 1.0, f"Reward {r} out of valid range [-1.0, 1.0]"


# ---------------------------------------------------------------------------
# TEST 5: TypeAdapter parses ALL 8 action types correctly
# ---------------------------------------------------------------------------

class TestActionParsingAllTypes:
    """Verifies that TypeAdapter(FaultLineAction).validate_python() works for every action."""

    VALID_PAYLOADS = [
        {
            "type": "query_logs",
            "service": "elasticsearch",
            "time_range": "last_15m",
            "filter_expr": "level=ERROR",
        },
        {
            "type": "check_metrics",
            "service": "elasticsearch",
            "metric_name": "memory",
            "window_minutes": 15,
        },
        {
            "type": "acknowledge_alert",
            "alert_id": "alert-001",
        },
        {
            "type": "rollback",
            "service": "payment-service",
            "target_version": "v1.2.3",
        },
        {
            "type": "scale_service",
            "service": "search-service",
            "replicas": 5,
        },
        {
            "type": "escalate",
            "team": "platform-sre",
            "message": "Cannot determine root cause within SLA",
        },
        {
            "type": "resolve",
            "root_cause_service": "elasticsearch",
            "postmortem_text": "OOM killed ES, causing search latency spike.",
        },
        {
            "type": "query_runbook",
            "topic": "oom",
        },
    ]

    @pytest.mark.parametrize("payload", VALID_PAYLOADS, ids=[p["type"] for p in VALID_PAYLOADS])
    def test_parse_action_type(self, payload):
        """Each action payload must parse without error and produce the correct Python type."""
        action = _ACTION_ADAPTER.validate_python(payload)
        assert action.type == payload["type"], (
            f"Parsed action type mismatch: expected {payload['type']!r}, got {action.type!r}"
        )

    def test_all_action_types_accepted_by_env(self):
        """All valid action types must be accepted by env.step() without raising."""
        env = FaultLineEnv("single_service_latency", seed=42)
        env.reset()

        runnable = [
            AcknowledgeAlertAction(alert_id="alert-001"),
            QueryLogsAction(service="elasticsearch", time_range="last_15m"),
            CheckMetricsAction(service="elasticsearch", metric_name="memory"),
            QueryRunbookAction(topic="oom"),
        ]
        for action in runnable:
            result = env.step(action)
            assert "reward" in result, f"Missing 'reward' for action type={action.type}"
            assert "observation" in result, f"Missing 'observation' for action type={action.type}"

        # Terminal action last
        result = env.step(EscalateAction(team="sre", message="Escalating"))
        assert result["done"] is True, "Escalate must set done=True"
