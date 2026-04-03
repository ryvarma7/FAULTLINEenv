from abc import ABC, abstractmethod
from typing import Any, Dict, List
from faultline.models import (
    Alert, FaultLineObservation, FaultLineAction, StepResult, EpisodeState
)

DEPENDENCY_GRAPH: Dict[str, List[str]] = {
    "api-gateway": ["auth-service", "product-service", "order-service"],
    "auth-service": ["user-db", "redis-cache"],
    "product-service": ["postgres-primary", "redis-cache"],
    "order-service": ["product-service", "payment-service", "postgres-primary"],
    "payment-service": ["payment-db", "fraud-detector"],
    "notification-service": ["order-service", "smtp-relay"],
    "postgres-primary": [],
    "redis-cache": [],
    "fraud-detector": ["model-serving", "payment-db"],
    "model-serving": [],
    "search-service": ["elasticsearch"],
    "elasticsearch": [],
}

class BaseTask(ABC):
    task_id: str
    difficulty: str
    max_steps: int = 20

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.episode_state = EpisodeState.ACTIVE
        self.elapsed_steps = 0
        self.cumulative_reward = 0.0
        self.acknowledged_alerts: List[str] = []
        self.log_query_counts: Dict[str, int] = {}
        self.reward_breakdown: Dict[str, float] = {}
        self._queried_root_cause_logs = False
        self._queried_anomaly_metrics: set = set()

    @abstractmethod
    def get_initial_observation(self) -> FaultLineObservation:
        """Return the starting observation for this task."""

    @abstractmethod
    def get_alerts(self) -> List[Alert]:
        """Return the list of alerts that fire at episode start."""

    @abstractmethod
    def get_root_cause_service(self) -> str:
        """Return the correct root cause service name."""

    @abstractmethod
    def get_correct_terminal_action(self) -> str:
        """Return the correct terminal action type: 'rollback', 'scale_service', or 'resolve'."""

    def is_done(self) -> bool:
        return self.episode_state != EpisodeState.ACTIVE or self.elapsed_steps >= self.max_steps

    def add_reward(self, key: str, value: float) -> None:
        if key not in self.reward_breakdown:
            self.reward_breakdown[key] = 0.0
        self.reward_breakdown[key] += value
        self.cumulative_reward = min(1.0, max(0.0, self.cumulative_reward + value))
