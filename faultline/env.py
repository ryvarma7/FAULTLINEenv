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
            
        return self._current_observation

    def step(self, action: FaultLineAction) -> dict:
        """
        Execute one action in the environment.
        Returns dict with observation, reward, done flag, and info dict.
        """
        if self._task is None and self._scenario is None:
            raise RuntimeError("Call reset() before step().")
            
        if self._task is not None:
            result = self._task.step(action)
            self._current_observation = result.observation
            return {
                "observation": result.observation,
                "reward": result.reward,
                "done": result.done,
                "info": result.info
            }
        else:
            # Simple handling for generated scenario
            done = False
            reward = 0.0
            if getattr(action, 'type', '') == self._scenario.correct_action_type:
                done = True
                reward = 1.0
            
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
