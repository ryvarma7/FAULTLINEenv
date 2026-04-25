"""
debug_sanity.py — FaultLine 5-Step Sanity Check
=================================================
Minimal health-check script. Run this before every training session.

Verifies:
  1. env.reset() works and returns alerts
  2. 5 steps execute without crashing
  3. At least one reward is non-zero (reward pipeline is alive)
  4. system_health_check() passes all 3 tasks

Usage:
  python debug_sanity.py
  python debug_sanity.py --task cascading_failure
  python debug_sanity.py --all

Exit codes:
  0 — all checks passed
  1 — reward pipeline broken (all rewards zero) or exception

DO NOT RUN THIS DURING TRAINING. It is a pre-flight check only.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from typing import Dict, List

from faultline.env import FaultLineEnv
from faultline.models import QueryLogsAction, AcknowledgeAlertAction, ResolveAction
from faultline.utils.action_parser import parse_action
from faultline.utils.validators import validate_step_output


# ---------------------------------------------------------------------------
# Per-task step sequence for health check
# ---------------------------------------------------------------------------

TASK_ACTIONS: Dict[str, list] = {
    "single_service_latency": [
        {"type": "query_logs", "service": "search-service", "time_range": "last_15m", "filter_expr": ""},
        {"type": "acknowledge_alert", "alert_id": "alert-001"},
        {"type": "query_logs", "service": "elasticsearch", "time_range": "last_15m", "filter_expr": ""},
        {"type": "check_metrics", "service": "elasticsearch", "metric_name": "memory"},
        {"type": "resolve", "root_cause_service": "elasticsearch",
         "postmortem_text": "Elasticsearch OOM due to GC pressure caused search latency."},
    ],
    "cascading_failure": [
        {"type": "query_logs", "service": "order-service", "time_range": "last_15m", "filter_expr": ""},
        {"type": "acknowledge_alert", "alert_id": "alert-001"},
        {"type": "query_logs", "service": "payment-service", "time_range": "last_15m", "filter_expr": ""},
        {"type": "check_metrics", "service": "payment-service", "metric_name": "error_rate"},
        {"type": "rollback", "service": "payment-service", "target_version": "v1.9.1"},
    ],
    "multi_region_incident": [
        {"type": "query_logs", "service": "api-gateway", "time_range": "last_15m", "filter_expr": ""},
        {"type": "acknowledge_alert", "alert_id": "alert-001"},
        {"type": "query_logs", "service": "model-serving", "time_range": "last_15m", "filter_expr": ""},
        {"type": "check_metrics", "service": "model-serving", "metric_name": "cpu"},
        {"type": "scale_service", "service": "model-serving", "replicas": 8},
    ],
}


def system_health_check(task_id: str = "single_service_latency", seed: int = 42) -> bool:
    """
    Execute a 5-step episode and assert the reward pipeline is alive.

    Returns:
        True if at least one reward was non-zero.
        False if all rewards are 0.0 (reward pipeline broken).

    FAIL if all rewards == 0.
    """
    print(f"\n[SANITY] Running health check: task={task_id} seed={seed}", flush=True)

    try:
        env = FaultLineEnv(task_id=task_id, seed=seed)
        obs = env.reset(task_id=task_id, seed=seed)
    except Exception as e:
        print(f"[SANITY][FAIL] env.reset() raised: {e}", flush=True)
        traceback.print_exc()
        return False

    print(f"[SANITY] Reset OK. Alerts: {len(obs.alerts)}", flush=True)
    for a in obs.alerts:
        print(f"  [{a.severity}] {a.id} | {a.service} | {a.title}", flush=True)

    rewards: List[float] = []
    actions = TASK_ACTIONS.get(task_id, TASK_ACTIONS["single_service_latency"])

    for i, action_dict in enumerate(actions, 1):
        # Parse via centralized parser
        action, error = parse_action(action_dict)
        if error:
            print(f"[SANITY][WARN] Step {i}: action parse failed: {error}", flush=True)
            rewards.append(0.0)
            continue

        # Execute step
        try:
            result = env.step(action)
            # Validate contract immediately
            validate_step_output(result)
        except Exception as e:
            print(f"[SANITY][FAIL] Step {i}: env.step() raised or contract violated: {e}", flush=True)
            traceback.print_exc()
            rewards.append(0.0)
            continue

        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        rewards.append(reward)

        obs_obj = result.get("observation")
        last_result = ""
        if obs_obj:
            if hasattr(obs_obj, "last_action_result"):
                last_result = obs_obj.last_action_result
            elif isinstance(obs_obj, dict):
                last_result = obs_obj.get("last_action_result", "")

        print(
            f"[SANITY] Step {i:2d} | {action.type:<22} | reward={reward:+.4f} | done={done}",
            flush=True,
        )
        if last_result:
            print(f"          +-- {last_result[:80]}", flush=True)

        if done:
            print(f"[SANITY] Episode ended at step {i}.", flush=True)
            break

    # ----------------------------------------------------------------
    # Final assertion: reward pipeline must produce at least one non-zero
    # ----------------------------------------------------------------
    total = sum(rewards)
    any_nonzero = any(r != 0.0 for r in rewards)

    print(f"\n[SANITY] Rewards   : {[f'{r:+.4f}' for r in rewards]}", flush=True)
    print(f"[SANITY] Total     : {total:+.4f}", flush=True)
    print(f"[SANITY] Non-zero  : {sum(1 for r in rewards if r != 0.0)} / {len(rewards)}", flush=True)

    if any_nonzero:
        print(f"[SANITY][PASS] Reward pipeline is ALIVE for task={task_id}.", flush=True)
    else:
        print(
            f"[SANITY][FAIL] ALL REWARDS ARE ZERO for task={task_id}. "
            f"Reward pipeline is BROKEN. DO NOT START TRAINING.",
            flush=True,
        )

    # Hard assertion — this will crash if reward pipeline is dead
    assert any_nonzero, (
        f"System broken: all rewards zero for task={task_id}. rewards={rewards}"
    )

    return any_nonzero


def main():
    parser = argparse.ArgumentParser(description="FaultLine 5-Step Sanity Check")
    parser.add_argument(
        "--task",
        default="single_service_latency",
        choices=list(TASK_ACTIONS),
        help="Task to run (default: single_service_latency)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--all", action="store_true", help="Run all 3 tasks")
    args = parser.parse_args()

    if args.all:
        results: Dict[str, bool] = {}
        for task_id in TASK_ACTIONS:
            try:
                results[task_id] = system_health_check(task_id=task_id, seed=args.seed)
            except AssertionError as e:
                print(f"[SANITY][FAIL] {task_id}: {e}", flush=True)
                results[task_id] = False

        print("\n" + "=" * 60, flush=True)
        print("  SANITY CHECK SUMMARY", flush=True)
        print("=" * 60, flush=True)
        all_passed = True
        for task_id, passed in results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} | {task_id}", flush=True)
            if not passed:
                all_passed = False

        if all_passed:
            print("\n  ALL SYSTEMS GO. Training is safe to start.", flush=True)
        else:
            print("\n  SYSTEM NOT READY. Fix failing tasks before training.", flush=True)

        sys.exit(0 if all_passed else 1)

    else:
        try:
            passed = system_health_check(task_id=args.task, seed=args.seed)
            sys.exit(0 if passed else 1)
        except AssertionError as e:
            print(f"[SANITY][FAIL] {e}", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
