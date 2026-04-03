from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Alert(BaseModel):
    id: str  # e.g. "alert-001"
    severity: str  # "P1", "P2", "P3", "P4"
    service: str  # service name e.g. "search-service"
    title: str
    description: str
    firing_since: str  # ISO timestamp string
    related_alerts: List[str]  # list of related alert IDs


class LogEntry(BaseModel):
    timestamp: str
    level: str  # "ERROR", "WARN", "INFO", "DEBUG"
    service: str
    trace_id: str
    message: str


class MetricPoint(BaseModel):
    timestamp: str
    value: float


class MetricSeries(BaseModel):
    service: str
    metric_name: str
    window_minutes: int
    points: List[MetricPoint]


class FaultLineObservation(BaseModel):
    alerts: List[Alert]
    log_results: Optional[List[LogEntry]] = None
    metric_results: Optional[MetricSeries] = None
    dependency_graph: Dict[str, List[str]]
    acknowledged_alerts: List[str] = []
    elapsed_steps: int = 0
    last_action_result: str = ""


class QueryLogsAction(BaseModel):
    type: Literal["query_logs"] = "query_logs"
    service: str
    time_range: str  # e.g. "last_15m"
    filter_expr: str = ""  # optional filter string


class CheckMetricsAction(BaseModel):
    type: Literal["check_metrics"] = "check_metrics"
    service: str
    metric_name: str  # "cpu", "memory", "latency_p99", "error_rate", "throughput"
    window_minutes: int = 15


class AcknowledgeAlertAction(BaseModel):
    type: Literal["acknowledge_alert"] = "acknowledge_alert"
    alert_id: str


class RollbackAction(BaseModel):
    type: Literal["rollback"] = "rollback"
    service: str
    target_version: str


class ScaleServiceAction(BaseModel):
    type: Literal["scale_service"] = "scale_service"
    service: str
    replicas: int


class EscalateAction(BaseModel):
    type: Literal["escalate"] = "escalate"
    team: str
    message: str


class ResolveAction(BaseModel):
    type: Literal["resolve"] = "resolve"
    root_cause_service: str
    postmortem_text: str


FaultLineAction = Annotated[
    Union[
        QueryLogsAction,
        CheckMetricsAction,
        AcknowledgeAlertAction,
        RollbackAction,
        ScaleServiceAction,
        EscalateAction,
        ResolveAction,
    ],
    Field(discriminator="type"),
]


class FaultLineReward(BaseModel):
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    reward_breakdown: Dict[str, float] = {}


class StepResult(BaseModel):
    observation: FaultLineObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class EpisodeState(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    FAILED = "failed"
    ESCALATED = "escalated"
