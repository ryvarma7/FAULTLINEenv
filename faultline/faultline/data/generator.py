import json
import random
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from faultline.models import Alert, LogEntry, MetricPoint, MetricSeries

DATA_DIR = Path(__file__).parent

def load_json(filename: str) -> dict:
    with open(DATA_DIR / filename) as f:
        return json.load(f)

def generate_logs(service: str, scenario: str, seed: int, count: int = 8) -> List[LogEntry]:
    """Generate synthetic log entries for a service/scenario using seed for reproducibility."""
    rng = random.Random(seed)
    templates_data = load_json("log_templates.json")
    service_templates = templates_data.get(service, {})
    templates = service_templates.get(scenario, service_templates.get("NORMAL", ["Log entry for {service}"]))
    
    entries = []
    base_time = datetime(2024, 1, 15, 14, 30, 0)
    for i in range(count):
        template = rng.choice(templates)
        message = template.format(
            heap_used=rng.randint(3500, 3900),
            heap_max=4096,
            heap_pct=rng.randint(90, 99),
            gc_time=rng.randint(800, 2000),
            shard_count=rng.randint(5, 20),
            index_name=f"logs-{rng.randint(1,9)}",
            latency=rng.randint(2100, 8000),
            timeout=rng.randint(2000, 5000),
            port=9200,
            query=rng.choice(["shoes", "laptop", "book"]),
            hit_rate=rng.randint(60, 90),
            active=rng.randint(95, 100),
            max=100,
            wait=rng.randint(5000, 30000),
            pool_size=rng.randint(10, 20),
            age=rng.randint(300, 3600),
            client=f"10.0.{rng.randint(1,255)}.{rng.randint(1,255)}",
            tx_id=f"tx-{rng.randint(10000,99999)}",
            pages=rng.randint(100, 5000),
            table=rng.choice(["payments", "orders", "users"]),
            mem_pct=rng.randint(85, 95),
            used=rng.randint(6800, 7500),
            total=8192,
            queue_depth=rng.randint(200, 600),
            error_count=rng.randint(50, 200),
            order_id=f"ord-{rng.randint(10000,99999)}",
            throttle_pct=rng.randint(80, 99),
            cpu_limit=2000,
            cpu_request=rng.randint(2100, 4000),
            consecutive_failures=rng.randint(5, 20),
            score=round(rng.uniform(0.1, 0.9), 2),
            version=f"v{rng.randint(1,5)}.{rng.randint(0,9)}.{rng.randint(0,9)}",
            service=service,
        )
        ts = (base_time + timedelta(seconds=i * rng.randint(5, 30))).isoformat() + "Z"
        level = "ERROR" if scenario not in ("NORMAL",) else rng.choice(["INFO", "WARN"])
        entries.append(LogEntry(
            timestamp=ts,
            level=level,
            service=service,
            trace_id=f"trace-{rng.randint(100000, 999999)}",
            message=message,
        ))
    return entries

def generate_metrics(service: str, metric_name: str, window_minutes: int, seed: int) -> MetricSeries:
    """Generate a synthetic time-series metric using seed for reproducibility."""
    rng = random.Random(seed + hash(service + metric_name))
    profiles_data = load_json("metric_profiles.json")
    
    service_metrics = profiles_data.get("service_metrics", {}).get(service, {})
    profile_name = service_metrics.get(metric_name, "steady")
    profile = profiles_data["profiles"].get(profile_name, profiles_data["profiles"]["steady"])
    
    base_value = profile["base_value"]
    noise_amp = profile["noise_amplitude"]
    spike = profile.get("spike")
    
    num_points = window_minutes
    base_time = datetime(2024, 1, 15, 14, 0, 0)
    points = []
    
    for i in range(num_points):
        pct = i / max(num_points - 1, 1)
        value = base_value + rng.uniform(-noise_amp, noise_amp)
        if spike:
            if pct >= spike["start_pct"]:
                progress = (pct - spike["start_pct"]) / max(spike["duration_pct"], 0.01)
                progress = min(progress, 1.0)
                value = base_value + (spike["peak_value"] - base_value) * progress
                value += rng.uniform(-noise_amp, noise_amp)
        value = max(0.0, min(100.0, value))
        ts = (base_time + timedelta(minutes=i)).isoformat() + "Z"
        points.append(MetricPoint(timestamp=ts, value=round(value, 2)))
    
    return MetricSeries(
        service=service,
        metric_name=metric_name,
        window_minutes=window_minutes,
        points=points,
    )
