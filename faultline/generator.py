import random
from typing import Optional
from faultline.models import IncidentConfig, IncidentScenario, Alert

class ProceduralIncidentGenerator:
    def generate(self, config: IncidentConfig, seed: int) -> IncidentScenario:
        rng = random.Random(seed)
        
        # 1. Determine root cause service based on failure mode
        services = ["frontend", "auth-service", "user-service", "cart-service", 
                    "payment-service", "inventory-service", "search-service", 
                    "recommendation-service", "database", "cache", "message-queue"]
        
        failure_map = {
            "latency": ["search-service", "recommendation-service", "database"],
            "crash": ["frontend", "auth-service", "payment-service"],
            "oom": ["cart-service", "inventory-service", "cache"],
            "config_drift": ["user-service", "payment-service"],
            "connection_leak": ["auth-service", "database", "message-queue"],
            "quota_exceeded": ["search-service", "recommendation-service"]
        }
        
        possible_roots = failure_map.get(config.failure_mode, services)
        root_cause_service = rng.choice(possible_roots)
        
        # 2. Determine affected services (cascade)
        affected_services = []
        available_cascade = [s for s in services if s != root_cause_service]
        rng.shuffle(available_cascade)
        for i in range(config.cascade_depth):
            if i < len(available_cascade):
                affected_services.append(available_cascade[i])
                
        # 3. Generate firing alerts
        firing_alerts = []
        alert_id_counter = 1
        
        # Root cause alert
        firing_alerts.append(Alert(
            id=f"alert-{alert_id_counter:03d}",
            severity="P1",
            service=root_cause_service,
            title=f"{config.failure_mode.capitalize()} issue detected",
            description=f"Critical {config.failure_mode} in {root_cause_service}",
            firing_since="2026-04-23T10:00:00Z",
            related_alerts=[]
        ))
        alert_id_counter += 1
        
        # Cascade alerts
        for svc in affected_services:
            firing_alerts.append(Alert(
                id=f"alert-{alert_id_counter:03d}",
                severity="P2",
                service=svc,
                title=f"Elevated error rate",
                description=f"Service {svc} failing due to upstream issues",
                firing_since="2026-04-23T10:02:00Z",
                related_alerts=[f"alert-001"]
            ))
            alert_id_counter += 1
            
        # 4. Generate red herring alerts
        red_herring_alerts = []
        available_herrings = [s for s in services if s not in affected_services and s != root_cause_service]
        rng.shuffle(available_herrings)
        
        for i in range(config.red_herring_count):
            if i < len(available_herrings):
                svc = available_herrings[i]
                red_herring_alerts.append(Alert(
                    id=f"alert-{alert_id_counter:03d}",
                    severity=rng.choice(["P3", "P4"]),
                    service=svc,
                    title="Minor anomaly detected",
                    description=f"Unrelated transient issue in {svc}",
                    firing_since="2026-04-23T09:30:00Z",
                    related_alerts=[]
                ))
                alert_id_counter += 1
                
        # 5. Determine correct action
        action_map = {
            "latency": "scale_service",
            "crash": "rollback",
            "oom": "rollback",
            "config_drift": "resolve",
            "connection_leak": "rollback",
            "quota_exceeded": "scale_service"
        }
        
        correct_action_type = action_map.get(config.failure_mode, "resolve")
        correct_version = "v1.2.3" if correct_action_type == "rollback" else None
        
        # 6. Keywords and logs
        trigger_keyword = f"{config.failure_mode}_error"
        remediation_verb = "fixed" if correct_action_type == "resolve" else correct_action_type
        
        log_templates = {
            root_cause_service: [f"ERROR {trigger_keyword} occurred", "WARN Retrying connection"],
        }
        for svc in affected_services:
            log_templates[svc] = ["ERROR Timeout waiting for response", "WARN Upstream degraded"]
            
        metric_anomaly_service = root_cause_service
        
        return IncidentScenario(
            root_cause_service=root_cause_service,
            affected_services=affected_services,
            firing_alerts=firing_alerts,
            red_herring_alerts=red_herring_alerts,
            correct_action_type=correct_action_type,
            correct_version=correct_version,
            trigger_keyword=trigger_keyword,
            remediation_verb=remediation_verb,
            log_templates=log_templates,
            metric_anomaly_service=metric_anomaly_service
        )
