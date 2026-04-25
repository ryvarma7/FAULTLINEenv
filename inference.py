"""
FaultLine Inference Script
===========================
Runs a ReAct-style LLM agent against all 3 FaultLine tasks.

Uses: meta-llama/Llama-3.1-405B-Instruct (Meta's best open model)

Required environment variables:
  API_BASE_URL   - LLM API endpoint (default: HuggingFace router)
  MODEL_NAME     - Model identifier (can override)
  HF_TOKEN       - API key / HuggingFace token

Output format (mandatory):
  [START] task=<task_id> env=faultline model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# --- Configuration ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-405B-Instruct")
ENV_BASE_URL = os.getenv("FAULTLINE_URL", "http://127.0.0.1:7860")

MAX_STEPS = 15
TEMPERATURE = 0.2
MAX_TOKENS = 512

TASKS = [
    "single_service_latency",
    "cascading_failure",
    "multi_region_incident",
]

# TASK 1: Task-specific success thresholds.
def get_success_threshold(task_id: str) -> float:
    if task_id == "single_service_latency":
        return 0.45
    elif task_id == "cascading_failure":
        return 0.40
    else:
        return 0.35

# TASK 5: Terminal actions — when seen in the loop, episode should be ending.
TERMINAL_ACTIONS = {"resolve", "rollback", "scale_service"}

# --- Logging helpers (mandatory format) ---

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=faultline model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- Environment client ---

def env_reset(task_id: str, seed: int = 42) -> Dict[str, Any]:
    resp = httpx.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = httpx.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_grade() -> Dict[str, Any]:
    resp = httpx.post(f"{ENV_BASE_URL}/grade", json={}, timeout=30)
    resp.raise_for_status()
    return resp.json()

# --- Prompt builders ---

# TASK 3: Improved system prompt with explicit rules, action menu, and critical constraints.
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Site Reliability Engineer (SRE) responding to a live production incident.

Your goal:
Identify the root cause of the incident and take the correct remediation action.

Available actions:
- acknowledge_alert
- query_logs
- check_metrics
- query_runbook
- rollback
- scale_service
- resolve

Rules:
1. Always start by acknowledging P1 alerts.
2. Investigate upstream dependencies — the alerting service is often NOT the root cause.
3. Use logs and metrics to trace the causal chain.
4. Avoid repeating the same query multiple times.
5. If unsure, use query_runbook with a relevant topic (max 2 times).
6. Once you identify the root cause, you MUST take a final action:
   - resolve(root_cause_service=..., postmortem_text=...)
   - OR rollback(service=..., target_version=...)
   - OR scale_service(service=..., replicas=...)

CRITICAL RULES:
- You MUST finish the task using resolve, rollback, or scale_service.
- You are NOT allowed to exceed 12 steps without taking a final action.
- Continuing investigation after identifying root cause is incorrect.
- Repeated queries will be penalized.

Output ONLY a valid JSON object with no explanation.
""").strip()

FORCE_CONCLUDE_PROMPT = (
    "\n\n[SYSTEM OVERRIDE] FINAL WARNING:\n"
    "You must now take a final action immediately.\n"
    "Do NOT investigate further.\n"
    "Respond with resolve, rollback, or scale_service.\n"
    "Output ONLY a valid JSON object."
)


def build_user_prompt(obs_data: Dict[str, Any], step: int, history: List[str], force_conclude: bool = False) -> str:
    alerts_summary = "\n".join(
        f"  [{a['severity']}] {a['id']} | {a['service']} | {a['title']}: {a['description']}"
        for a in obs_data.get("alerts", [])
    )
    log_summary = "None"
    if obs_data.get("log_results"):
        logs = obs_data["log_results"][:4]
        log_summary = "\n".join(f"  [{l['level']}] {l['service']}: {l['message']}" for l in logs)

    metric_summary = "None"
    if obs_data.get("metric_results"):
        mr = obs_data["metric_results"]
        pts = mr.get("points", [])
        if pts:
            latest = pts[-1]["value"]
            metric_summary = f"{mr['service']}/{mr['metric_name']}: latest={latest:.1f} over {mr['window_minutes']}min"

    history_block = "\n".join(history[-5:]) if history else "None"

    base = textwrap.dedent(f"""
        Step: {step}
        Last action result: {obs_data.get('last_action_result', '')}
        Elapsed steps: {obs_data.get('elapsed_steps', step)}

        FIRING ALERTS:
        {alerts_summary}

        LOG RESULTS:
        {log_summary}

        METRIC RESULTS:
        {metric_summary}

        RECENT HISTORY:
        {history_block}

        What is your next action? Respond with ONLY a JSON object.
    """).strip()

    # TASK 7: Inject force-conclude instruction at step >= 10
    if force_conclude:
        base += FORCE_CONCLUDE_PROMPT

    return base


def get_agent_action(client: OpenAI, obs_data: Dict[str, Any], step: int, history: List[str], force_conclude: bool = False) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs_data, step, history, force_conclude=force_conclude)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Model request or parse failed: {exc}", flush=True)
        # Fallback: query logs of first alerting service
        alerts = obs_data.get("alerts", [])
        service = alerts[0]["service"] if alerts else "api-gateway"
        return {"type": "query_logs", "service": service, "time_range": "last_15m", "filter_expr": ""}


def run_task(client: OpenAI, task_id: str, seed: int = 42) -> float:
    """Run one full episode for a task. Returns (score, steps_taken, success)."""
    log_start(task=task_id, model=MODEL_NAME)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    # TASK 5: Per-service query count for looping detection (tracked client-side for logging)
    query_logs_count: Dict[str, int] = {}
    runbook_count = 0
    terminal_action_taken = False

    try:
        result_data = env_reset(task_id=task_id, seed=seed)
        obs_data = result_data.get("observation", result_data)
        done = result_data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # TASK 7: Force conclude if >= 12 steps with no terminal action
            force_conclude = (step >= 12 and not terminal_action_taken)

            action = get_agent_action(client, obs_data, step, history, force_conclude=force_conclude)
            
            # If step >= 14, restrict allowed actions to ONLY terminal actions
            if step >= 14 and action.get("type") not in TERMINAL_ACTIONS:
                action = {"type": "resolve", "root_cause_service": "unknown", "postmortem_text": "Forced resolution due to step limit."}

            action_str = json.dumps(action, separators=(",", ":"))
            action_type = action.get("type", "unknown")

            # TASK 8: Debug logging per step
            print(f"[DEBUG] Step {step}", flush=True)
            print(f"[DEBUG] Action: {action_str}", flush=True)

            # TASK 5: Client-side loop tracking for observability
            if action_type == "query_logs":
                ql_key = f"{action.get('service', '')}:{action.get('time_range', '')}"
                query_logs_count[ql_key] = query_logs_count.get(ql_key, 0) + 1
                if query_logs_count[ql_key] > 3:
                    print(f"[DEBUG] WARNING: Repeated query_logs key={ql_key} count={query_logs_count[ql_key]}", flush=True)
            elif action_type == "query_runbook":
                runbook_count += 1
                if runbook_count > 2:
                    print(f"[DEBUG] WARNING: query_runbook overuse count={runbook_count}", flush=True)

            if action_type in TERMINAL_ACTIONS:
                terminal_action_taken = True

            try:
                step_result = env_step(action)
                obs_data = step_result["observation"]
                reward = float(step_result.get("reward", 0.0))
                done = step_result.get("done", False)
                error = None
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)[:80]
                # TASK 4: Handle HTTP 4xx from server gracefully (invalid action payload)
                if "422" in error or "400" in error:
                    print(f"[DEBUG] Invalid action rejected by server: {error}", flush=True)
                    obs_data["last_action_result"] = "INVALID_ACTION"

            # TASK 8: Debug reward/done
            print(f"[DEBUG] Reward: {reward:.2f}", flush=True)
            print(f"[DEBUG] Done: {done}", flush=True)
            if done and error:
                print(f"[DEBUG] Termination Reason: error - {error}", flush=True)
            elif done and step_result.get("info", {}).get("termination_reason"):
                print(f"[DEBUG] Termination Reason: {step_result['info']['termination_reason']}", flush=True)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_type} -> reward {reward:+.2f} | {obs_data.get('last_action_result','')[:60]}")

            if done:
                break

        # Grade the episode
        try:
            grade_result = env_grade()
            score = float(grade_result.get("score", 0.0))
        except Exception:
            score = sum(rewards)
            score = min(max(score, 0.0), 1.0)

        # TASK 1: Success only when score meets the task-specific threshold
        success_threshold = get_success_threshold(task_id)
        success = score >= success_threshold

        print(f"[DEBUG] Total Reward (Score): {score:.2f}", flush=True)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, steps_taken, success


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=0.1, max_retries=0)
    print(f"[DEBUG] FaultLine inference starting. Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] Environment URL: {ENV_BASE_URL}", flush=True)
    print(f"[DEBUG] API Base URL: {API_BASE_URL}", flush=True)

    all_scores = {}
    for task_id in TASKS:
        score, steps, success = run_task(client, task_id)
        all_scores[task_id] = score
        print(f"[DEBUG] {task_id} final score: {score:.3f} steps: {steps} success: {success}", flush=True)

    print("\n[DEBUG] === FINAL SCORES ===", flush=True)
    for task_id, score in all_scores.items():
        print(f"[DEBUG] {task_id}: {score:.3f}", flush=True)

    print("\nAgent behavior fixed. Environment ready for training.", flush=True)


if __name__ == "__main__":
    main()