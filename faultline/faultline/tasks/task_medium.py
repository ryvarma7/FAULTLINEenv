import json
from pathlib import Path
from typing import List
from faultline.models import (
    Alert, FaultLineObservation, FaultLineAction,
    QueryLogsAction, CheckMetricsAction, AcknowledgeAlertAction,
    RollbackAction, EscalateAction, StepResult, EpisodeState
)
from faultline.tasks.base import BaseTask, DEPENDENCY_GRAPH
from faultline.data.generator import generate_logs, generate_metrics

DATA_DIR = Path(__file__).parent.parent / "data"

class TaskMedium(BaseTask):
    task_id = "cascading_failure"
    difficulty = "medium"
    max_steps = 20

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        with open(DATA_DIR / "incidents" / "medium.json") as f:
            self.config = json.load(f)
        self._queried_payment_db_logs = False
        self._red_herring_queries: set = set()

    def get_alerts(self) -> List[Alert]:
        return [Alert(**a) for a in self.config["alerts"]]

    def get_root_cause_service(self) -> str:
        return self.config["root_cause_service"]

    def get_correct_terminal_action(self) -> str:
        return self.config["correct_action"]

    def get_initial_observation(self) -> FaultLineObservation:
        return FaultLineObservation(
            alerts=self.get_alerts(),
            log_results=None,
            metric_results=None,
            dependency_graph=DEPENDENCY_GRAPH,
            acknowledged_alerts=[],
            elapsed_steps=0,
            last_action_result="Incident detected. Alerts are firing. Begin investigation.",
        )

    def step(self, action: FaultLineAction) -> StepResult:
        """Process one action and return the resulting StepResult."""
        if self.is_done():
            obs = self.get_initial_observation()
            obs.elapsed_steps = self.elapsed_steps
            obs.acknowledged_alerts = self.acknowledged_alerts
            return StepResult(observation=obs, reward=0.0, done=True,
                              info={"episode_state": self.episode_state})

        self.elapsed_steps += 1
        step_reward = 0.0
        result_message = ""
        done = False

        if isinstance(action, AcknowledgeAlertAction):
            if action.alert_id not in self.acknowledged_alerts:
                self.acknowledged_alerts.append(action.alert_id)
                alert_obj = next((a for a in self.get_alerts() if a.id == action.alert_id), None)
                if alert_obj and alert_obj.severity == "P1":
                    step_reward += 0.10
                    self.add_reward("ack_p1_alert", 0.10)
                result_message = f"Alert {action.alert_id} acknowledged."
            else:
                result_message = f"Alert {action.alert_id} already acknowledged."

        elif isinstance(action, QueryLogsAction):
            key = f"{action.service}:{action.time_range}"
            self.log_query_counts[key] = self.log_query_counts.get(key, 0) + 1
            if self.log_query_counts[key] > 3:
                step_reward -= 0.05
                self.add_reward("loop_penalty", -0.05)
                result_message = f"Warning: repeated log query for {action.service}. Penalty applied."
            else:
                if action.service == "payment-db" and not self._queried_payment_db_logs:
                    step_reward += 0.10
                    self.add_reward("queried_payment_db_logs", 0.10)
                    self._queried_payment_db_logs = True
                
                # Determine anomaly type based on service
                if action.service == "payment-service":
                    anomaly_type = "CONNECTION_POOL_LEAK"
                elif action.service == "payment-db":
                    anomaly_type = "CONNECTION_EXHAUSTED"
                elif action.service == "order-service":
                    anomaly_type = "HIGH_MEMORY"
                else:
                    anomaly_type = "NORMAL"
                
                logs = generate_logs(action.service, anomaly_type, self.seed)
                result_message = f"Retrieved {len(logs)} log entries for {action.service}."
            
            obs = self._build_obs(result_message, log_results=generate_logs(
                action.service,
                "CONNECTION_POOL_LEAK" if action.service == "payment-service" else
                "CONNECTION_EXHAUSTED" if action.service == "payment-db" else
                "HIGH_MEMORY" if action.service == "order-service" else "NORMAL",
                self.seed
            ))
            self.cumulative_reward = min(1.0, max(0.0, self.cumulative_reward + step_reward))
            return StepResult(observation=obs, reward=step_reward, done=False,
                              info={"episode_state": self.episode_state, "reward_breakdown": self.reward_breakdown})

        elif isinstance(action, CheckMetricsAction):
            metric_key = f"{action.service}:{action.metric_name}"
            anomalous_combos = {
                ("payment-service", "error_rate"),
                ("payment-service", "latency_p99"),
            }
            if (action.service, action.metric_name) in anomalous_combos and metric_key not in self._queried_anomaly_metrics:
                step_reward += 0.05
                self.add_reward(f"metric_anomaly_{metric_key}", 0.05)
                self._queried_anomaly_metrics.add(metric_key)
            
            metrics = generate_metrics(action.service, action.metric_name, action.window_minutes, self.seed)
            result_message = f"Metrics for {action.service}/{action.metric_name}: latest={metrics.points[-1].value:.1f}"
            obs = self._build_obs(result_message, metric_results=metrics)
            self.cumulative_reward = min(1.0, max(0.0, self.cumulative_reward + step_reward))
            return StepResult(observation=obs, reward=step_reward, done=False,
                              info={"episode_state": self.episode_state, "reward_breakdown": self.reward_breakdown})

        elif isinstance(action, RollbackAction):
            done = True
            if action.service == self.get_root_cause_service():
                step_reward += 0.40
                self.add_reward("correct_rollback", 0.40)
                self.episode_state = EpisodeState.RESOLVED
                if self.elapsed_steps <= 12:
                    step_reward += 0.10
                    self.add_reward("speed_bonus", 0.10)
            else:
                step_reward -= 0.20
                self.add_reward("wrong_rollback_target", -0.20)
                self.episode_state = EpisodeState.FAILED
            result_message = f"Rollback action submitted for service: {action.service}."

        elif isinstance(action, EscalateAction):
            done = True
            self.episode_state = EpisodeState.ESCALATED
            step_reward += 0.05
            self.add_reward("escalation", 0.05)
            result_message = f"Escalated to {action.team}. Human team paged."

        else:
            result_message = "Action type not supported in this task."

        self.cumulative_reward = min(1.0, max(0.0, self.cumulative_reward + step_reward))
        obs = self._build_obs(result_message)
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=done or self.is_done(),
            info={"episode_state": self.episode_state, "reward_breakdown": self.reward_breakdown},
        )

    def _build_obs(self, result_message: str, log_results=None, metric_results=None) -> FaultLineObservation:
        return FaultLineObservation(
            alerts=self.get_alerts(),
            log_results=log_results,
            metric_results=metric_results,
            dependency_graph=DEPENDENCY_GRAPH,
            acknowledged_alerts=self.acknowledged_alerts.copy(),
            elapsed_steps=self.elapsed_steps,
            last_action_result=result_message,
        )
