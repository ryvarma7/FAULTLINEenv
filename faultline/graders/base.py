from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List
from faultline.tasks.base import BaseTask

@dataclass
class GraderResult:
    score: float                           # 0.0 to 1.0, clamped
    breakdown: Dict[str, float] = field(default_factory=dict)
    passed: bool = False
    notes: List[str] = field(default_factory=list)

class BaseGrader(ABC):
    """Deterministic grader that scores a completed episode."""

    @abstractmethod
    def grade(self, task: BaseTask) -> GraderResult:
        """Grade the completed task episode. Returns GraderResult."""

    def _clamp(self, value: float) -> float:
        val = max(0.0, min(1.0, round(value, 4)))
        if val <= 0.0:
            return 0.01
        elif val >= 1.0:
            return 0.99
        return val
