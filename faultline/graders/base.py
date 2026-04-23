from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List
from faultline.tasks.base import BaseTask
from typing import Any

def score_postmortem(text: str, scenario: Any) -> float:
    score = 0.0
    text_lower = text.lower()
    
    root_cause = ""
    trigger = ""
    remediation = ""
    
    if hasattr(scenario, 'root_cause_service'):
        root_cause = scenario.root_cause_service
        trigger = getattr(scenario, 'trigger_keyword', '')
        remediation = getattr(scenario, 'remediation_verb', '')
    elif hasattr(scenario, 'get_root_cause_service'):
        root_cause = scenario.get_root_cause_service()
        keywords = getattr(scenario, 'config', {}).get("postmortem_keywords", [])
        if len(keywords) > 0:
            trigger = keywords[0]
        if len(keywords) > 1:
            remediation = keywords[1]

    if root_cause and root_cause.lower() in text_lower:
        score += 0.05
    if trigger and trigger.lower() in text_lower:
        score += 0.05
    if remediation and remediation.lower() in text_lower:
        score += 0.05
        
    return round(score, 3)

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
