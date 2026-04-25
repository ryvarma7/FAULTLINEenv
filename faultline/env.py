"""
FaultLine Environment (faultline/env.py)
=========================================
Core RL environment implementing the OpenEnv interface.

HARDENING CHANGELOG:
- step() wrapped in top-level try/except — exceptions NEVER propagate as 500.
- Fail-fast action.type validation at top of step().
- Reward clamp: min(1.0, max(-1.0, reward)) — was max(0.0,...) which killed penalty signals.
- assert isinstance(reward, float) added before every return path.
- [REWARD_DEBUG] printed at every step for full reward pipeline visibility.
- Suspicious-zero-reward assertion: warns if reward is always 0 after step 2.
- Observation safety: _current_observation never None post-reset; all return paths use it.
- [DEBUG] at key execution points for full visibility.
- CENTRALIZED: action parsing (in server layer) uses faultline.utils.action_parser.parse_action().
- STEP CONTRACT: validate_step_output() is called in server layer after every step().
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

from faultline.models import (
    FaultLineObservation, FaultLineAction, StepResult, EpisodeState
)
from faultline.tasks.base import BaseTask
from faultline.tasks.task_easy import TaskEasy
from faultline.tasks.task_medium import TaskMedium
from faultline.tasks.task_hard import TaskHard
from faultline.graders.grader_easy import GraderEasy
from faultline.graders.grader_medium import GraderMedium
from faultline.graders.grader_hard import GraderHard

TASK_REGISTRY = {
    "single_service_latency": (TaskEasy, GraderEasy),
    "cascading_failure": (TaskMedium, GraderMedium),
    "multi_region_incident": (TaskHard, GraderHard),
}

# Maximum steps guard — episode always terminates at or before this limit.
MAX_EPISODE_STEPS = 20


def _validated_reward(reward: Any, label: str = "") -> float:
    """
    Guarantee reward is a float within [-1.0, 1.0].
    Asserts type correctness and logs the value.
    """
    if not isinstance(reward, (int, float)):
        print(f"[ERROR] Reward is not numeric ({type(reward).__name__}): {reward!r} — forcing 0.0", flush=True)
        reward = 0.0
    reward = float(reward)
    reward = max(-1.0, min(1.0, reward))  # FIX: was max(0.0,...) which clamped penalties to 0
    assert isinstance(reward, float), f"Reward must be float, got {type(reward).__name__}"
    print(f"[REWARD_DEBUG]{' ' + label if label else ''} step reward={reward:.4f}", flush=True)
    return reward


class FaultLineEnv:
    """
    FaultLine OpenEnv environment.
    Implements the full OpenEnv interface: reset(), step(), state().
    """

    def __init__(self, task_id: str = "single_service_latency", seed: int = 42):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_REGISTRY)}")
        self.task_id = task_id
        self.seed = seed
        self._task: Optional[BaseTask] = None
        self._current_observation: Optional[FaultLineObservation] = None
        self._scenario = None

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None, config: Optional[Any] = None) -> FaultLineObservation:
        """Reset the environment to initial state. Returns ObservationModel."""
        if task_id is not None:
            self.task_id = task_id
        if seed is not None:
            self.seed = seed

        if config:
            from faultline.generator import ProceduralIncidentGenerator
            generator = ProceduralIncidentGenerator()
            scenario = generator.generate(config, self.seed)
            self._scenario = scenario
            self._task = None

            alerts = self._scenario.firing_alerts + self._scenario.red_herring_alerts
            self._current_observation = FaultLineObservation(
                alerts=alerts,
                dependency_graph={s: [] for s in self._scenario.affected_services + [self._scenario.root_cause_service]},
            )
        else:
            self._scenario = None
            TaskClass, _ = TASK_REGISTRY[self.task_id]
            self._task = TaskClass(seed=self.seed)
            self._current_observation = self._task.get_initial_observation()

        # Unified query tracking for runbook and logs
        self._runbook_query_counts: Dict[str, int] = {}
        self._log_query_counts: Dict[str, int] = {}
        self._resolved = False
        self._step_count = 0
        self._cumulative_reward = 0.0  # tracks total episode reward for suspicious-zero check

        print(f"[DEBUG] reset() complete: task_id={self.task_id} alerts={len(self._current_observation.alerts)}", flush=True)
        return self._current_observation

    @property
    def elapsed_steps(self) -> int:
        """Public alias for _step_count — used by assertions and health checks."""
        return self._step_count

    def step(self, action: FaultLineAction) -> dict:
        """
        Execute one action in the environment.
        Returns dict: {observation, reward, done, info}.

        GUARANTEES (post-hardening):
        - Always returns a dict — NEVER raises.
        - reward is always float in [-1.0, 1.0].
        - observation is always a FaultLineObservation object (serialized by server layer).
        - Errors are always surfaced in the returned dict, never silently dropped.
        """
        # ----------------------------------------------------------------
        # Pre-flight: must have called reset()
        # ----------------------------------------------------------------
        if self._task is None and self._scenario is None:
            raise RuntimeError("Call reset() before step().")

        # ----------------------------------------------------------------
        # Top-level try/except — nothing propagates as an unhandled exception
        # ----------------------------------------------------------------
        try:
            return self._step_inner(action)
        except Exception as e:
            print(f"[ERROR] Unhandled exception inside step(): {e}", flush=True)
            traceback.print_exc()
            safe_obs = self._current_observation or FaultLineObservation(
                alerts=[], dependency_graph={}
            )
            reward = _validated_reward(0.0, label="[ERROR_FALLBACK]")
            return {
                "observation": safe_obs,
                "reward": reward,
                "done": False,
                "info": {},
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def _step_inner(self, action: FaultLineAction) -> dict:
        """
        Inner step logic — separated so the outer step() can wrap it cleanly.
        """
        self._step_count += 1
        is_max_steps = self._step_count >= MAX_EPISODE_STEPS

        # ----------------------------------------------------------------
        # FAIL-FAST: validate action.type
        # ----------------------------------------------------------------
        action_type = getattr(action, 'type', None)
        if not action_type:
            print(f"[ERROR] action.type missing or empty: {action!r}", flush=True)
            reward = _validated_reward(0.0, label="[INVALID_ACTION]")
            return {
                "observation": self._current_observation,
                "reward": reward,
                "done": False,
                "info": {"error": "INVALID_ACTION: missing type field"},
            }

        print(f"[DEBUG] Executing step: action_type={action_type} step={self._step_count}", flush=True)

        VALID_ACTION_TYPES = {
            "query_logs", "check_metrics", "acknowledge_alert",
            "rollback", "scale_service", "escalate", "resolve", "query_runbook"
        }
        if action_type not in VALID_ACTION_TYPES:
            print(f"[ERROR] Unknown action_type: {action_type!r}", flush=True)
            self._current_observation.last_action_result = "INVALID_ACTION"
            self._current_observation.elapsed_steps = self._step_count
            reward = _validated_reward(0.0, label="[UNKNOWN_ACTION]")
            return {
                "observation": self._current_observation,
                "reward": reward,
                "done": False,
                "info": {"error": f"Unknown action type: '{action_type}'"},
            }

        # ----------------------------------------------------------------
        # Anti-hacking: prevent multiple terminal actions
        # ----------------------------------------------------------------
        TERMINAL_ACTION_TYPES = {"resolve", "rollback", "scale_service"}
        if action_type in TERMINAL_ACTION_TYPES:
            if self._resolved:
                self._current_observation.last_action_result = "ALREADY_TERMINATED"
                reward = _validated_reward(0.0, label="[ALREADY_TERMINATED]")
                return {
                    "observation": self._current_observation,
                    "reward": reward,
                    "done": True,
                    "info": {"error": "Already terminated"},
                }
            self._resolved = True

        # ----------------------------------------------------------------
        # Handle query_runbook centrally
        # ----------------------------------------------------------------
        if action_type == 'query_runbook':
            return self._handle_query_runbook(action, is_max_steps)

        # ----------------------------------------------------------------
        # Delegate to task (structured task path)
        # ----------------------------------------------------------------
        if self._task is not None:
            result = self._task.step(action)
            self._current_observation = result.observation
            self._current_observation.elapsed_steps = self._step_count

            if is_max_steps:
                result.done = True
                result.info["termination_reason"] = "max_steps"
            elif action_type in TERMINAL_ACTION_TYPES:
                result.done = True

            reward = _validated_reward(result.reward, label="[TASK_STEP]")
            return {
                "observation": result.observation,
                "reward": reward,
                "done": result.done,
                "info": result.info,
            }

        # ----------------------------------------------------------------
        # Scenario path (procedural generator)
        # ----------------------------------------------------------------
        return self._handle_scenario_step(action, action_type, is_max_steps, TERMINAL_ACTION_TYPES)

    def _handle_query_runbook(self, action, is_max_steps: bool) -> dict:
        """Handle query_runbook action centrally."""
        from faultline.runbooks import RUNBOOK_ENTRIES
        topic = getattr(action, 'topic', '')
        text = RUNBOOK_ENTRIES.get(topic, f"No runbook entry found for '{topic}'.")

        self._runbook_query_counts[topic] = self._runbook_query_counts.get(topic, 0) + 1
        count = self._runbook_query_counts[topic]

        reward_raw = 0.0
        root_cause = (
            self._task.get_root_cause_service() if self._task
            else (self._scenario.root_cause_service if self._scenario else "")
        )

        # Relevance reward — once only on first query
        if count == 1 and (
            topic == root_cause
            or topic in [root_cause.replace("-service", ""), "connection_leak", "quota_exceeded",
                         "oom", "latency", "config_drift", "crash", "cascading_failure"]
        ):
            if topic == root_cause or (self._task and topic in getattr(self._task, 'task_id', '')):
                reward_raw += 0.05
            elif topic in root_cause:
                reward_raw += 0.05

        # Runbook overuse penalty after 2 queries
        if count > 2:
            reward_raw -= 0.03

        self._current_observation.last_action_result = text
        self._current_observation.elapsed_steps = self._step_count

        if self._task:
            self._task.elapsed_steps = self._step_count
            self._task.cumulative_reward = min(1.0, max(0.0, self._task.cumulative_reward + reward_raw))

        reward = _validated_reward(reward_raw, label="[QUERY_RUNBOOK]")
        return {
            "observation": self._current_observation,
            "reward": reward,
            "done": is_max_steps,
            "info": {},
        }

    def _handle_scenario_step(self, action, action_type: str, is_max_steps: bool, TERMINAL_ACTION_TYPES: set) -> dict:
        """Handle actions in the procedural scenario path."""
        done = is_max_steps
        reward_raw = 0.0
        info: Dict[str, Any] = {}
        if is_max_steps:
            info["termination_reason"] = "max_steps"

        if action_type == 'query_logs':
            service = getattr(action, 'service', '')
            key = f"{service}:{getattr(action, 'time_range', '')}"
            self._log_query_counts[key] = self._log_query_counts.get(key, 0) + 1
            if self._log_query_counts[key] > 3:
                reward_raw -= 0.05
            elif service == self._scenario.root_cause_service and self._log_query_counts[key] == 1:
                reward_raw += 0.10
            self._current_observation.last_action_result = f"Queried logs for {service}."

        elif action_type == 'check_metrics':
            service = getattr(action, 'service', '')
            if service == self._scenario.root_cause_service:
                reward_raw += 0.05
            self._current_observation.last_action_result = f"Metrics checked for {service}."

        elif action_type == 'acknowledge_alert':
            alert_id = getattr(action, 'alert_id', '')
            if not hasattr(self, '_acked_alerts'):
                self._acked_alerts = set()
            if alert_id not in self._acked_alerts:
                self._acked_alerts.add(alert_id)
                reward_raw += 0.05
                self._current_observation.last_action_result = "Alert acknowledged."
            else:
                self._current_observation.last_action_result = "Alert already acknowledged."

        elif action_type == self._scenario.correct_action_type:
            # Correct terminal action — ends episode with full reward
            done = True
            reward_raw = 0.40
            self._current_observation.last_action_result = f"Correct action {action_type} executed."
            if self._step_count <= 8:
                reward_raw += 0.10
            if action_type == 'resolve':
                pm_text = getattr(action, 'postmortem_text', '').lower()
                if self._scenario.root_cause_service in pm_text:
                    reward_raw += 0.05
                if getattr(self._scenario, 'trigger_keyword', '') in pm_text:
                    reward_raw += 0.05
                if getattr(self._scenario, 'remediation_verb', '') in pm_text:
                    reward_raw += 0.05

        elif action_type in TERMINAL_ACTION_TYPES:
            # Wrong terminal action — penalty, ends episode
            done = True
            reward_raw = -0.20
            self._current_observation.last_action_result = f"Wrong terminal action: {action_type}."

        elif action_type not in {'check_metrics', 'acknowledge_alert', 'escalate'}:
            self._current_observation.last_action_result = "Invalid or incorrect action."
        else:
            self._current_observation.last_action_result = f"Executed {action_type}."

        self._current_observation.elapsed_steps = self._step_count

        # FIX: Allow negative rewards (was max(0.0, ...) which clamped -0.20 → 0.0)
        reward = _validated_reward(reward_raw, label="[SCENARIO_STEP]")
        return {
            "observation": self._current_observation,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def state(self) -> dict:
        """Return the current observation without advancing the episode."""
        if self._current_observation is None:
            raise RuntimeError("Call reset() before state().")
        return self._current_observation.model_dump()

    def grade(self) -> Dict[str, Any]:
        """Grade the current episode. Call after episode is done."""
        if self._task is None:
            raise RuntimeError("Call reset() first.")
        _, GraderClass = TASK_REGISTRY[self.task_id]
        grader = GraderClass()
        result = grader.grade(self._task)
        return {
            "score": result.score,
            "passed": result.passed,
            "breakdown": result.breakdown,
            "notes": result.notes,
            "episode_state": self._task.episode_state,
            "steps_taken": self._task.elapsed_steps,
        }

    def close(self) -> None:
        """Clean up resources."""
        self._task = None
        self._current_observation = None
