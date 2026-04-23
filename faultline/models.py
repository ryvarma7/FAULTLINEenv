from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class FaultLineModel(BaseModel):
    # Compliance fix: explicit Pydantic v2 model configuration style.
    model_config = ConfigDict(extra="forbid")


class Alert(FaultLineModel):
    id: str  # e.g. "alert-001"
    severity: str  # "P1", "P2", "P3", "P4"
    service: str  # service name e.g. "search-service"
    title: str
    description: str
    firing_since: str  # ISO timestamp string
    related_alerts: List[str]  # list of related alert IDs


class LogEntry(FaultLineModel):
    timestamp: str
    level: str  # "ERROR", "WARN", "INFO", "DEBUG"
    service: str
    trace_id: str
    message: str


class MetricPoint(FaultLineModel):
    timestamp: str
    value: float


class MetricSeries(FaultLineModel):
    service: str
    metric_name: str
    window_minutes: int
    points: List[MetricPoint]


class FaultLineObservation(FaultLineModel):
    alerts: List[Alert]
    log_results: Optional[List[LogEntry]] = None
    metric_results: Optional[MetricSeries] = None
    dependency_graph: Dict[str, List[str]]
    acknowledged_alerts: List[str] = Field(default_factory=list)
    elapsed_steps: int = 0
    last_action_result: str = ""


class QueryLogsAction(FaultLineModel):
    type: Literal["query_logs"] = "query_logs"
    service: str
    time_range: str  # e.g. "last_15m"
    filter_expr: str = ""  # optional filter string


class CheckMetricsAction(FaultLineModel):
    type: Literal["check_metrics"] = "check_metrics"
    service: str
    metric_name: str  # "cpu", "memory", "latency_p99", "error_rate", "throughput"
    window_minutes: int = 15


class AcknowledgeAlertAction(FaultLineModel):
    type: Literal["acknowledge_alert"] = "acknowledge_alert"
    alert_id: str


class RollbackAction(FaultLineModel):
    type: Literal["rollback"] = "rollback"
    service: str
    target_version: str


class ScaleServiceAction(FaultLineModel):
    type: Literal["scale_service"] = "scale_service"
    service: str
    replicas: int


class EscalateAction(FaultLineModel):
    type: Literal["escalate"] = "escalate"
    team: str
    message: str


class ResolveAction(FaultLineModel):
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


class FaultLineReward(FaultLineModel):
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)


class StepResult(FaultLineModel):
    observation: FaultLineObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class IncidentConfig(FaultLineModel):
    failure_mode: Literal[
        "latency",
        "crash",
        "oom",
        "config_drift",
        "connection_leak",
        "quota_exceeded",
    ]
    cascade_depth: int = Field(ge=0, le=3)
    red_herring_count: int = Field(ge=0, le=3)
    noise_level: float = Field(ge=0.0, le=1.0)
    multi_region: bool = False


class IncidentScenario(FaultLineModel):
    root_cause_service: str
    affected_services: List[str]
    firing_alerts: List[Alert]
    red_herring_alerts: List[Alert]
    correct_action_type: Literal["resolve", "rollback", "scale_service"]
    correct_version: Optional[str] = None
    trigger_keyword: str
    remediation_verb: str
    log_templates: Dict[str, List[str]]
    metric_anomaly_service: str


class EpisodeState(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    FAILED = "failed"
    ESCALATED = "escalated"
