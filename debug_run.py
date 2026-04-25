"""
debug_run.py — FaultLine Manual Sanity Check Script
=====================================================
Runs a local episode directly against FaultLineEnv (no HTTP server required).

Actions executed:
  1. acknowledge_alert  — expect reward > 0
  2. query_logs (root cause service) — expect reward > 0
  3. check_metrics (anomalous service) — expect reward > 0
  4. query_runbook — small relevance reward
  5. resolve (correct root cause) — expect large reward > 0

Final assertion: at least one reward must be non-zero.

Usage:
  python debug_run.py
  python debug_run.py --task cascading_failure
  python debug_run.py --task multi_region_incident
"""
from __future__ import annotations

import argparse
import sys
from typing import List

from faultline.env import FaultLineEnv
from faultline.models import (
    AcknowledgeAlertAction,
    CheckMetricsAction,
    EscalateAction,
    QueryLogsAction,
    QueryRunbookAction,
    ResolveAction,
    RollbackAction,
    ScaleServiceAction,
)
from pydantic import TypeAdapter
from faultline.models import FaultLineAction

_ACTION_ADAPTER: TypeAdapter[FaultLineAction] = TypeAdapter(FaultLineAction)


# ---------------------------------------------------------------------------
# Per-task configuration
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "single_service_latency": {
        "alert_id": "alert-001",
        "root_cause_service": "elasticsearch",
        "log_service": "elasticsearch",
        "metric_service": "elasticsearch",
        "metric_name": "memory",
        "runbook_topic": "oom",
        "correct_action": ResolveAction(
            root_cause_service="elasticsearch",
            postmortem_text=(
                "Elasticsearch hit OOM due to GC pressure. "
                "Heap exhaustion caused search-service latency spike. "
                "Remediation: increased JVM heap, disabled query cache."
            ),
        ),
    },
    "cascading_failure": {
        "alert_id": "alert-001",
        "root_cause_service": "payment-service",
        "log_service": "payment-service",
        "metric_service": "payment-service",
        "metric_name": "error_rate",
        "runbook_topic": "connection_leak",
        "correct_action": RollbackAction(
            service="payment-service",
            target_version="v1.9.1",
        ),
    },
    "multi_region_incident": {
        "alert_id": "alert-001",
        "root_cause_service": "model-serving",
        "log_service": "model-serving",
        "metric_service": "model-serving",
        "metric_name": "cpu",
        "runbook_topic": "quota_exceeded",
        "correct_action": ScaleServiceAction(
            service="model-serving",
            replicas=8,
        ),
    },
}


def banner(msg: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{'='*60}", flush=True)


def print_result(step: int, action_type: str, result: dict) -> None:
    reward = result.get("reward", "MISSING")
    done = result.get("done", "MISSING")
    obs = result.get("observation")
    last_result = ""
    if obs:
        if hasattr(obs, "last_action_result"):
            last_result = obs.last_action_result
        elif isinstance(obs, dict):
            last_result = obs.get("last_action_result", "")
    error = result.get("error", "")
    error_str = f"  [ERROR]: {error}" if error else ""

    print(f"  Step {step:2d} | {action_type:<22} | reward={reward:>+7.4f} | done={done}{error_str}", flush=True)
    if last_result:
        print(f"           +-- obs: {last_result[:80]}", flush=True)


def run_sanity_check(task_id: str = "single_service_latency", seed: int = 42) -> bool:
    """
    Run a full sanity episode. Returns True if at least one reward was non-zero.
    """
    if task_id not in TASK_CONFIGS:
        print(f"[ERROR] Unknown task_id '{task_id}'. Choose from: {list(TASK_CONFIGS)}", flush=True)
        return False

    cfg = TASK_CONFIGS[task_id]
    banner(f"FaultLine Debug Run -- task={task_id} seed={seed}")

    # ----------------------------------------------------------------
    # 1. Reset
    # ----------------------------------------------------------------
    env = FaultLineEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    print(f"\n[RESET] Observation: {len(obs.alerts)} alert(s) firing", flush=True)
    for a in obs.alerts:
        print(f"  [{a.severity}] {a.id} | {a.service} | {a.title}", flush=True)

    # ----------------------------------------------------------------
    # 2. Test TypeAdapter parsing for all action types
    # ----------------------------------------------------------------
    banner("TypeAdapter Parsing -- All 8 Action Types")
    all_payloads = [
        {"type": "query_logs", "service": "elasticsearch", "time_range": "last_15m", "filter_expr": ""},
        {"type": "check_metrics", "service": "elasticsearch", "metric_name": "memory"},
        {"type": "acknowledge_alert", "alert_id": "alert-001"},
        {"type": "rollback", "service": "payment-service", "target_version": "v1.9.1"},
        {"type": "scale_service", "service": "model-serving", "replicas": 5},
        {"type": "escalate", "team": "sre-oncall", "message": "Escalating"},
        {"type": "resolve", "root_cause_service": "elasticsearch", "postmortem_text": "OOM"},
        {"type": "query_runbook", "topic": "oom"},
    ]
    parse_ok = True
    for payload in all_payloads:
        try:
            action = _ACTION_ADAPTER.validate_python(payload)
            print(f"  [OK]   {payload['type']:<22} -> {type(action).__name__}", flush=True)
        except Exception as e:
            print(f"  [FAIL] {payload['type']:<22} -> FAILED: {e}", flush=True)
            parse_ok = False

    if not parse_ok:
        print("\n[FATAL] TypeAdapter parsing is broken -- fix models.py", flush=True)
        return False

    # ----------------------------------------------------------------
    # 3. Episode steps
    # ----------------------------------------------------------------
    banner("Episode Steps")
    rewards: List[float] = []
    step = 0

    actions = [
        ("acknowledge_alert",  AcknowledgeAlertAction(alert_id=cfg["alert_id"])),
        ("query_logs",         QueryLogsAction(service=cfg["log_service"], time_range="last_15m")),
        ("check_metrics",      CheckMetricsAction(service=cfg["metric_service"], metric_name=cfg["metric_name"])),
        ("query_runbook",      QueryRunbookAction(topic=cfg["runbook_topic"])),
        ("terminal_action",    cfg["correct_action"]),
    ]

    done = False
    for action_label, action in actions:
        if done:
            print(f"  (episode already done, skipping {action_label})", flush=True)
            break
        step += 1
        result = env.step(action)
        reward = float(result.get("reward", 0.0))
        rewards.append(reward)
        done = bool(result.get("done", False))
        print_result(step, action_label, result)

    # ----------------------------------------------------------------
    # 4. Summary
    # ----------------------------------------------------------------
    banner("Summary")
    print(f"  Steps taken : {step}", flush=True)
    print(f"  Rewards     : {[f'{r:+.4f}' for r in rewards]}", flush=True)
    print(f"  Total reward: {sum(rewards):+.4f}", flush=True)
    print(f"  Non-zero    : {sum(1 for r in rewards if r != 0.0)} / {len(rewards)}", flush=True)

    any_nonzero = any(r != 0.0 for r in rewards)
    if any_nonzero:
        print("\n  [PASS] At least one non-zero reward observed.", flush=True)
    else:
        print("\n  [FAIL] All rewards are 0.0. Reward pipeline is broken!", flush=True)

    # ----------------------------------------------------------------
    # 5. Grade
    # ----------------------------------------------------------------
    try:
        grade = env.grade()
        print(f"\n  Grade: score={grade['score']:.3f} passed={grade['passed']}", flush=True)
        print(f"  Breakdown: {grade['breakdown']}", flush=True)
    except Exception as e:
        print(f"\n  Grade unavailable (scenario path): {e}", flush=True)

    return any_nonzero


def main():
    parser = argparse.ArgumentParser(description="FaultLine Debug Sanity Check")
    parser.add_argument("--task", default="single_service_latency",
                        choices=list(TASK_CONFIGS), help="Task ID to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--all", action="store_true", help="Run all 3 tasks")
    args = parser.parse_args()

    if args.all:
        results = {}
        for task_id in TASK_CONFIGS:
            results[task_id] = run_sanity_check(task_id, seed=args.seed)
        banner("ALL TASKS SUMMARY")
        all_pass = True
        for task_id, passed in results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} | {task_id}", flush=True)
            if not passed:
                all_pass = False
        sys.exit(0 if all_pass else 1)
    else:
        passed = run_sanity_check(task_id=args.task, seed=args.seed)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
