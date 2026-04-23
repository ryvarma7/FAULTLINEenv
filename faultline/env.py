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
            
        self._runbook_query_counts = {}
        self._resolved = False
        self._step_count = 0
            
        return self._current_observation

    def step(self, action: FaultLineAction) -> dict:
        """
        Execute one action in the environment.
        Returns dict with observation, reward, done flag, and info dict.
        """
        if self._task is None and self._scenario is None:
            raise RuntimeError("Call reset() before step().")
            
        self._step_count += 1
        is_max_steps = self._step_count >= 20

        action_type = getattr(action, 'type', '')
        
        # Anti-hacking: prevent repeated resolve
        if action_type == 'resolve':
            if self._resolved:
                self._current_observation.last_action_result = "Error: Episode already resolved or failed."
                return {
                    "observation": self._current_observation,
                    "reward": -0.1,
                    "done": True,
                    "info": {"error": "Already resolved"}
                }
            self._resolved = True

        # Handle QueryRunbookAction centrally
        if action_type == 'query_runbook':
            from faultline.runbooks import RUNBOOK_ENTRIES
            topic = getattr(action, 'topic', '')
            text = RUNBOOK_ENTRIES.get(topic, f"No runbook entry found for '{topic}'.")
            
            self._runbook_query_counts[topic] = self._runbook_query_counts.get(topic, 0) + 1
            count = self._runbook_query_counts[topic]
            
            reward = 0.0
            root_cause = self._task.get_root_cause_service() if self._task else (self._scenario.root_cause_service if self._scenario else "")
            
            if topic == root_cause or topic in [root_cause.replace("-service", ""), "connection_leak", "quota_exceeded", "oom", "latency", "config_drift", "crash", "cascading_failure"]:
                # simple relevance check
                if topic == root_cause or topic in getattr(self._scenario, 'failure_mode', '') or (self._task and topic in getattr(self._task, 'task_id', '')):
                    reward += 0.05
                elif topic in root_cause:
                    reward += 0.05
                    
            if count >= 3:
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
            if is_max_steps:
                result.done = True
            return {
                "observation": result.observation,
                "reward": result.reward,
                "done": result.done,
                "info": result.info
            }
        else:
            # Simple handling for generated scenario
            done = is_max_steps
            reward = 0.0
            
            if action_type == 'query_logs':
                # Anti-hacking query_logs spam penalty
                key = f"{getattr(action, 'service', '')}:{getattr(action, 'time_range', '')}"
                self._runbook_query_counts[key] = self._runbook_query_counts.get(key, 0) + 1
                if self._runbook_query_counts[key] > 3:
                    reward -= 0.05
                self._current_observation.last_action_result = f"Queried logs for {getattr(action, 'service', '')}."
            elif action_type == self._scenario.correct_action_type:
                done = True
                reward = 1.0
                self._current_observation.last_action_result = f"Correct action {action_type} executed."
            elif action_type not in ['check_metrics', 'acknowledge_alert', 'escalate']:
                self._current_observation.last_action_result = "Invalid or incorrect action."
            else:
                self._current_observation.last_action_result = f"Executed {action_type}."
                
            self._current_observation.elapsed_steps = self._step_count
            
            return {
                "observation": self._current_observation,
                "reward": reward,
                "done": done,
                "info": {}
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
