from __future__ import annotations
import json
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

# TASK 2: Maximum steps guard — episode always terminates at or before this limit.
MAX_EPISODE_STEPS = 20

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

            # Dummy initial observation for the generated scenario
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

        # TASK 5: Unified query tracking for runbook and logs
        self._runbook_query_counts: Dict[str, int] = {}
        self._log_query_counts: Dict[str, int] = {}
        self._resolved = False
        self._step_count = 0

        return self._current_observation

    def step(self, action: FaultLineAction) -> dict:
        """
        Execute one action in the environment.
        Returns dict with observation, reward, done flag, and info dict.

        TASK 2: Terminal actions (resolve, rollback, scale_service) ALWAYS set done=True.
                 Reaching max_steps also forces done=True.
        TASK 4: Invalid/unknown actions return reward=0.0, done=False, INVALID_ACTION message.
        TASK 5: Repeated query_logs (>3 per key) → -0.05 penalty.
                 Repeated query_runbook (>2 total) → -0.03 penalty.
        TASK 6: Correct terminal actions earn +0.40 only when root cause matches.
        """
        if self._task is None and self._scenario is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1
        # TASK 2: Force done when max steps reached
        is_max_steps = self._step_count >= MAX_EPISODE_STEPS

        action_type = getattr(action, 'type', '')

        # TASK 4: Guard against unsupported action types gracefully
        VALID_ACTION_TYPES = {
            "query_logs", "check_metrics", "acknowledge_alert",
            "rollback", "scale_service", "escalate", "resolve", "query_runbook"
        }
        if action_type not in VALID_ACTION_TYPES:
            self._current_observation.last_action_result = "INVALID_ACTION"
            self._current_observation.elapsed_steps = self._step_count
            return {
                "observation": self._current_observation,
                "reward": 0.0,
                "done": False,
                "info": {"error": f"Unknown action type: '{action_type}'"}
            }

        # Anti-hacking: prevent multiple terminal actions
        TERMINAL_ACTION_TYPES = {"resolve", "rollback", "scale_service"}
        if action_type in TERMINAL_ACTION_TYPES:
            if self._resolved:
                self._current_observation.last_action_result = "ALREADY_TERMINATED"
                return {
                    "observation": self._current_observation,
                    "reward": 0.0,
                    "done": True,
                    "info": {"error": "Already terminated"}
                }
            self._resolved = True

        # TASK 2: Terminal actions ALWAYS set done=True
        TERMINAL_ACTION_TYPES = {"resolve", "rollback", "scale_service"}

        # Handle QueryRunbookAction centrally
        if action_type == 'query_runbook':
            from faultline.runbooks import RUNBOOK_ENTRIES
            topic = getattr(action, 'topic', '')
            text = RUNBOOK_ENTRIES.get(topic, f"No runbook entry found for '{topic}'.")

            self._runbook_query_counts[topic] = self._runbook_query_counts.get(topic, 0) + 1
            count = self._runbook_query_counts[topic]

            reward = 0.0
            root_cause = (
                self._task.get_root_cause_service() if self._task
                else (self._scenario.root_cause_service if self._scenario else "")
            )

            # Relevance reward (once only, first query)
            if count == 1 and (
                topic == root_cause
                or topic in [root_cause.replace("-service", ""), "connection_leak", "quota_exceeded", "oom", "latency", "config_drift", "crash", "cascading_failure"]
            ):
                if topic == root_cause or (self._task and topic in getattr(self._task, 'task_id', '')):
                    reward += 0.05
                elif topic in root_cause:
                    reward += 0.05

            # TASK 5: Runbook overuse penalty after 2 queries
            if count > 2:
                reward -= 0.03

            self._current_observation.last_action_result = text
            self._current_observation.elapsed_steps = self._step_count

            if self._task:
                self._task.elapsed_steps = self._step_count
                self._task.cumulative_reward = min(1.0, max(0.0, self._task.cumulative_reward + reward))

            return {
                "observation": self._current_observation,
                "reward": reward,
                "done": is_max_steps,
                "info": {}
            }

        if self._task is not None:
            result = self._task.step(action)
            self._current_observation = result.observation
            self._current_observation.elapsed_steps = self._step_count

            # TASK 2: Force done on max_steps OR if a terminal action was taken
            if is_max_steps:
                result.done = True
                result.info["termination_reason"] = "max_steps"
            elif action_type in TERMINAL_ACTION_TYPES:
                result.done = True

            return {
                "observation": result.observation,
                "reward": result.reward,
                "done": result.done,
                "info": result.info
            }
        else:
            done = False
            if is_max_steps:
                done = True

            reward = 0.0
            info = {}
            if is_max_steps:
                info["termination_reason"] = "max_steps"

            if action_type == 'query_logs':
                # TASK 5: Anti-hacking query_logs spam penalty
                service = getattr(action, 'service', '')
                key = f"{service}:{getattr(action, 'time_range', '')}"
                self._log_query_counts[key] = self._log_query_counts.get(key, 0) + 1
                if self._log_query_counts[key] > 3:
                    reward -= 0.05
                elif service == self._scenario.root_cause_service and self._log_query_counts[key] == 1:
                    reward += 0.10
                self._current_observation.last_action_result = f"Queried logs for {getattr(action, 'service', '')}."

            elif action_type == 'check_metrics':
                service = getattr(action, 'service', '')
                if service == self._scenario.root_cause_service:
                    reward += 0.05
                self._current_observation.last_action_result = f"Metrics checked for {service}."

            elif action_type == 'acknowledge_alert':
                # First time only ack
                alert_id = getattr(action, 'alert_id', '')
                if not hasattr(self, '_acked_alerts'):
                    self._acked_alerts = set()
                if alert_id not in self._acked_alerts:
                    self._acked_alerts.add(alert_id)
                    reward += 0.05
                    self._current_observation.last_action_result = "Alert acknowledged."
                else:
                    self._current_observation.last_action_result = "Alert already acknowledged."

            elif action_type == self._scenario.correct_action_type:
                # TASK 2 + 6: Correct terminal action ends episode with full reward
                done = True
                reward = 0.40
                self._current_observation.last_action_result = f"Correct action {action_type} executed."
                if self._step_count <= 8:
                    reward += 0.10
                if action_type == 'resolve':
                    pm_text = getattr(action, 'postmortem_text', '').lower()
                    if self._scenario.root_cause_service in pm_text:
                        reward += 0.05
                    if getattr(self._scenario, 'trigger_keyword', '') in pm_text:
                        reward += 0.05
                    if getattr(self._scenario, 'remediation_verb', '') in pm_text:
                        reward += 0.05

            elif action_type in TERMINAL_ACTION_TYPES:
                # TASK 6: Wrong terminal action gets penalty and ends episode
                done = True
                reward = -0.20
                self._current_observation.last_action_result = f"Wrong terminal action: {action_type}."

            elif action_type not in {'check_metrics', 'acknowledge_alert', 'escalate'}:
                self._current_observation.last_action_result = "Invalid or incorrect action."
            else:
                self._current_observation.last_action_result = f"Executed {action_type}."

            self._current_observation.elapsed_steps = self._step_count

            # Clamp total reward
            reward = min(1.0, max(0.0, reward))

            return {
                "observation": self._current_observation,
                "reward": reward,
                "done": done,
                "info": info
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
        """Clean up resources (no-op for this environment)."""
        self._task = None
        self._current_observation = None
