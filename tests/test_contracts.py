"""
tests/test_contracts.py
=========================
Permanent contract tests for the FaultLine hardening system.

These tests exercise THREE layers simultaneously:
  1. faultline.utils.action_parser.parse_action()  — centralized parsing
  2. faultline.utils.validators.validate_step_output() — step contract
  3. FaultLineEnv.step()  — end-to-end reward pipeline

GUARANTEES UNDER TEST:
  - Action parsing NEVER crashes — always returns (action, None) or (None, error)
  - step() ALWAYS returns {observation, reward, done}
  - reward pipeline is NOT always zero
  - invalid input NEVER breaks the system
  - step() contract is enforced on every call

Run with:
  pytest tests/test_contracts.py -v
"""
from __future__ import annotations

import pytest

from faultline.env import FaultLineEnv
from faultline.models import (
    AcknowledgeAlertAction,
    CheckMetricsAction,
    QueryLogsAction,
    QueryRunbookAction,
    ResolveAction,
    RollbackAction,
    ScaleServiceAction,
    EscalateAction,
)
from faultline.utils.action_parser import parse_action
from faultline.utils.validators import validate_step_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh FaultLineEnv, reset and ready for each test."""
    e = FaultLineEnv("single_service_latency", seed=42)
    e.reset()
    return e


# ---------------------------------------------------------------------------
# 1. test_action_parsing_valid
#    parse_action() must succeed for all 8 action types.
# ---------------------------------------------------------------------------

class TestActionParsingValid:
    """parse_action() must return (action, None) for all valid payloads."""

    VALID_PAYLOADS = [
        {"type": "query_logs", "service": "elasticsearch", "time_range": "last_15m", "filter_expr": ""},
        {"type": "check_metrics", "service": "elasticsearch", "metric_name": "memory", "window_minutes": 15},
        {"type": "acknowledge_alert", "alert_id": "alert-001"},
        {"type": "rollback", "service": "payment-service", "target_version": "v1.9.1"},
        {"type": "scale_service", "service": "model-serving", "replicas": 5},
        {"type": "escalate", "team": "sre-oncall", "message": "Cannot determine root cause"},
        {"type": "resolve", "root_cause_service": "elasticsearch", "postmortem_text": "OOM in ES."},
        {"type": "query_runbook", "topic": "oom"},
    ]

    @pytest.mark.parametrize("payload", VALID_PAYLOADS, ids=[p["type"] for p in VALID_PAYLOADS])
    def test_action_parsing_valid(self, payload):
        """Every valid action payload must parse to (action, None) — no error."""
        action, error = parse_action(payload)

        assert error is None, (
            f"Expected no error for payload {payload['type']!r}, "
            f"got error: {error}"
        )
        assert action is not None, f"Expected action object, got None for {payload['type']!r}"
        assert action.type == payload["type"], (
            f"Type mismatch: expected {payload['type']!r}, got {action.type!r}"
        )


# ---------------------------------------------------------------------------
# 2. test_action_parsing_invalid
#    parse_action() must return (None, error_dict) for all invalid payloads.
#    It must NEVER raise.
# ---------------------------------------------------------------------------

class TestActionParsingInvalid:
    """parse_action() must return (None, error) for all invalid payloads."""

    INVALID_PAYLOADS = [
        {},                                                  # empty dict
        {"type": "nuke_datacenter"},                         # unknown type
        {"type": "query_logs"},                              # missing required fields
        {"type": "resolve"},                                 # missing root_cause_service + postmortem_text
        {"type": "rollback", "service": "svc"},             # missing target_version
        {"not_a_type": "query_logs"},                        # no type field
        None,                                                # None
        "not a dict",                                        # wrong type entirely
        {"type": "query_logs", "extra_field": "bad"},        # extra field (extra='forbid')
    ]

    @pytest.mark.parametrize("payload", INVALID_PAYLOADS)
    def test_action_parsing_invalid(self, payload):
        """Invalid payloads must return (None, error_dict) — NEVER raise."""
        try:
            action, error = parse_action(payload)
        except Exception as exc:
            pytest.fail(
                f"parse_action() raised an exception instead of returning (None, error): {exc!r}"
            )

        assert action is None, f"Expected action=None for invalid payload, got {action!r}"
        assert error is not None, "Expected an error dict, got None"
        assert isinstance(error, dict), f"Error must be a dict, got {type(error).__name__}"
        assert "error" in error, f"Error dict missing 'error' key: {error}"
        assert "message" in error, f"Error dict missing 'message' key: {error}"


# ---------------------------------------------------------------------------
# 3. test_step_returns_valid_dict
#    env.step() must ALWAYS return {observation, reward, done}.
#    validate_step_output() must pass for all valid actions.
# ---------------------------------------------------------------------------

class TestStepReturnsValidDict:
    """env.step() must always return a contract-compliant dict."""

    def _assert_step_contract(self, result: dict) -> None:
        """Helper: assert step result satisfies the OpenEnv contract."""
        # validate_step_output raises if broken — this surfaces bugs immediately
        validate_step_output(result)

        assert isinstance(result["reward"], float), (
            f"reward must be float, got {type(result['reward']).__name__}"
        )
        assert isinstance(result["done"], bool), (
            f"done must be bool, got {type(result['done']).__name__}"
        )
        assert result["observation"] is not None, "observation must not be None"

    def test_acknowledge_alert_contract(self, env):
        result = env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        self._assert_step_contract(result)

    def test_query_logs_contract(self, env):
        result = env.step(QueryLogsAction(service="elasticsearch", time_range="last_15m"))
        self._assert_step_contract(result)

    def test_check_metrics_contract(self, env):
        result = env.step(CheckMetricsAction(service="elasticsearch", metric_name="memory"))
        self._assert_step_contract(result)

    def test_resolve_contract(self, env):
        result = env.step(ResolveAction(root_cause_service="elasticsearch", postmortem_text="OOM"))
        self._assert_step_contract(result)

    def test_escalate_contract(self, env):
        result = env.step(EscalateAction(team="sre-oncall", message="Cannot determine root cause"))
        self._assert_step_contract(result)

    def test_query_runbook_contract(self, env):
        result = env.step(QueryRunbookAction(topic="oom"))
        self._assert_step_contract(result)

    def test_rollback_contract(self, env):
        result = env.step(RollbackAction(service="elasticsearch", target_version="v1.2.3"))
        self._assert_step_contract(result)

    def test_scale_service_contract(self, env):
        result = env.step(ScaleServiceAction(service="elasticsearch", replicas=3))
        self._assert_step_contract(result)


# ---------------------------------------------------------------------------
# 4. test_reward_not_always_zero
#    At least one step in a standard episode must produce reward != 0.
#    If ALL rewards are 0.0, the reward pipeline is broken.
# ---------------------------------------------------------------------------

class TestRewardNotAlwaysZero:
    """Reward pipeline must produce non-zero signals during a normal episode."""

    def test_reward_not_always_zero(self, env):
        """At least one reward must be non-zero during a 5-step episode."""
        actions = [
            AcknowledgeAlertAction(alert_id="alert-001"),
            QueryLogsAction(service="elasticsearch", time_range="last_15m"),
            CheckMetricsAction(service="elasticsearch", metric_name="memory"),
            QueryRunbookAction(topic="oom"),
            ResolveAction(
                root_cause_service="elasticsearch",
                postmortem_text=(
                    "Elasticsearch hit OOM due to GC pressure. "
                    "Heap exhaustion caused search-service latency spike. "
                    "Remediation: increased JVM heap, disabled query cache."
                ),
            ),
        ]

        rewards = []
        for action in actions:
            result = env.step(action)
            rewards.append(result["reward"])
            if result["done"]:
                break

        assert any(r != 0.0 for r in rewards), (
            f"REWARD PIPELINE BROKEN: all rewards were 0.0 across {len(rewards)} steps. "
            f"rewards={rewards}"
        )

    def test_correct_resolve_reward_positive(self, env):
        """Correct resolve must give reward > 0."""
        result = env.step(ResolveAction(
            root_cause_service="elasticsearch",
            postmortem_text="OOM in elasticsearch caused search latency.",
        ))
        assert result["reward"] > 0.0, (
            f"Correct resolve should give reward > 0, got {result['reward']}"
        )

    def test_acknowledge_p1_reward_positive(self, env):
        """Acknowledging a P1/P2 alert must give reward > 0."""
        result = env.step(AcknowledgeAlertAction(alert_id="alert-001"))
        assert result["reward"] > 0.0, (
            f"Expected reward > 0 for P1/P2 ack, got {result['reward']}"
        )

    def test_reward_range_valid_across_episode(self, env):
        """Every reward must be within [-1.0, 1.0]."""
        actions = [
            AcknowledgeAlertAction(alert_id="alert-001"),
            QueryLogsAction(service="elasticsearch", time_range="last_15m"),
            CheckMetricsAction(service="elasticsearch", metric_name="memory"),
        ]
        for action in actions:
            result = env.step(action)
            r = result["reward"]
            assert -1.0 <= r <= 1.0, (
                f"Reward {r} is outside valid range [-1.0, 1.0]"
            )


# ---------------------------------------------------------------------------
# 5. test_invalid_action_does_not_crash
#    Sending invalid/garbage actions must NEVER crash the system.
# ---------------------------------------------------------------------------

class TestInvalidActionDoesNotCrash:
    """Invalid actions must be handled gracefully at all layers."""

    def test_parse_action_never_raises_on_garbage(self):
        """parse_action() must handle arbitrary garbage without raising."""
        garbage_inputs = [
            None,
            "",
            123,
            [],
            {"type": ""},
            {"type": "UNKNOWN_TYPE", "data": "irrelevant"},
            {"service": "elasticsearch"},  # missing type
        ]
        for inp in garbage_inputs:
            try:
                action, error = parse_action(inp)
                # Must always return a tuple, never raise
                assert action is None or error is None, \
                    "Must return exactly one of (action, None) or (None, error)"
            except Exception as exc:
                pytest.fail(
                    f"parse_action() raised on garbage input {inp!r}: {exc!r}"
                )

    def test_env_step_with_monkey_patched_invalid_type(self, env):
        """
        If an action with an unknown type reaches env.step() (e.g. bypass), it
        must return a safe dict — not raise.
        """
        # Construct a valid action then monkey-patch to simulate internal bypass
        action = QueryLogsAction(service="api-gateway", time_range="last_15m")
        object.__setattr__(action, "type", "totally_invalid_type")

        result = env.step(action)  # MUST NOT raise

        assert isinstance(result, dict), "step() must return a dict even for patched-invalid actions"
        assert "reward" in result, "step() must always include 'reward'"
        assert "done" in result, "step() must always include 'done'"
        assert "observation" in result, "step() must always include 'observation'"
        assert result["reward"] == 0.0, "Unknown action type must return reward=0.0"
        assert result["done"] is False, "Unknown action type must return done=False"

    def test_env_handles_missing_fields_gracefully(self):
        """
        Parsing an action with missing required fields must not break the env.
        The error should be caught at parse_action(), not inside env.step().
        """
        action, error = parse_action({"type": "resolve"})  # missing required fields
        assert action is None
        assert error is not None
        assert error["error"] == "ACTION_PARSE_ERROR"

    def test_validate_step_output_raises_on_bad_dict(self):
        """validate_step_output must raise clearly for each contract violation."""
        # Missing 'reward'
        with pytest.raises(ValueError, match="reward"):
            validate_step_output({"observation": {}, "done": False})

        # Missing 'done'
        with pytest.raises(ValueError, match="done"):
            validate_step_output({"observation": {}, "reward": 0.0})

        # Missing 'observation'
        with pytest.raises(ValueError, match="observation"):
            validate_step_output({"reward": 0.0, "done": False})

        # Non-numeric reward
        with pytest.raises(TypeError, match="numeric"):
            validate_step_output({"observation": {}, "reward": "bad", "done": False})

        # Non-bool done
        with pytest.raises(TypeError, match="bool"):
            validate_step_output({"observation": {}, "reward": 0.0, "done": "yes"})

        # None observation
        with pytest.raises(ValueError, match="None"):
            validate_step_output({"observation": None, "reward": 0.0, "done": False})
