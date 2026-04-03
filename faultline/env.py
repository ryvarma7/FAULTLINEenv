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

    def reset(self) -> StepResult:
        """Reset the environment to initial state. Returns initial StepResult."""
        TaskClass, _ = TASK_REGISTRY[self.task_id]
        self._task = TaskClass(seed=self.seed)
        self._current_observation = self._task.get_initial_observation()
        return StepResult(
            observation=self._current_observation,
            reward=0.0,
            done=False,
            info={
                "task_id": self.task_id,
                "seed": self.seed,
                "episode_state": EpisodeState.ACTIVE,
            },
        )

    def step(self, action: FaultLineAction) -> StepResult:
        """
        Execute one action in the environment.
        Returns StepResult with updated observation, reward, done flag, and info dict.
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        result = self._task.step(action)
        self._current_observation = result.observation
        return result

    def state(self) -> FaultLineObservation:
        """Return the current observation without advancing the episode."""
        if self._current_observation is None:
            raise RuntimeError("Call reset() before state().")
        return self._current_observation

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
