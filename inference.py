"""
FaultLine Inference Script
===========================
Runs a ReAct-style LLM agent against all 3 FaultLine tasks.

Required environment variables:
  API_BASE_URL   - LLM API endpoint (default: HuggingFace router)
  MODEL_NAME     - Model identifier
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
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("FAULTLINE_URL", "http://localhost:7860")

MAX_STEPS = 15
TEMPERATURE = 0.2
MAX_TOKENS = 512

TASKS = [
    "single_service_latency",
    "cascading_failure",
    "multi_region_incident",
]

SUCCESS_THRESHOLD = 0.10

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

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Site Reliability Engineer (SRE) responding to a live production incident.
You will receive an observation containing:
- alerts: list of firing alerts (id, severity P1-P4, service, title, description)
- dependency_graph: service dependency map
- log_results: logs from your last query (if any)
- metric_results: metrics from your last query (if any)
- last_action_result: result of your last action
- elapsed_steps: steps taken so far

You must respond with ONLY a JSON object representing ONE action. Choose from:

{"type": "acknowledge_alert", "alert_id": "<alert_id>"}
{"type": "query_logs", "service": "<service_name>", "time_range": "last_15m", "filter_expr": ""}
{"type": "check_metrics", "service": "<service_name>", "metric_name": "<cpu|memory|latency_p99|error_rate|throughput>", "window_minutes": 15}
{"type": "rollback", "service": "<service_name>", "target_version": "<version>"}
{"type": "scale_service", "service": "<service_name>", "replicas": <int>}
{"type": "escalate", "team": "<team_name>", "message": "<message>"}
{"type": "resolve", "root_cause_service": "<service_name>", "postmortem_text": "<detailed postmortem>"}

Strategy:
1. Acknowledge P1 alerts first.
2. Check metrics and logs to trace the root cause upstream.
3. Do NOT query the same service/range more than twice.
4. Ignore low-severity (P3/P4) alerts unless they directly relate to P1s.
5. When confident, call resolve() with the correct root_cause_service and a detailed postmortem.
6. Postmortem must mention: root cause, timeline, and remediation steps.

Respond with ONLY the JSON object, no other text.
""").strip()

def build_user_prompt(obs_data: Dict[str, Any], step: int, history: List[str]) -> str:
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

    return textwrap.dedent(f"""
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

def get_agent_action(client: OpenAI, obs_data: Dict[str, Any], step: int, history: List[str]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs_data, step, history)
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

def run_task(client: OpenAI, task_id: str) -> float:
    """Run one full episode for a task. Returns normalized score."""
    log_start(task=task_id, model=MODEL_NAME)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result_data = env_reset(task_id=task_id, seed=42)
        obs_data = result_data["observation"]
        done = result_data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_agent_action(client, obs_data, step, history)
            action_str = json.dumps(action, separators=(",", ":"))

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

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action.get('type')} -> reward {reward:+.2f} | {obs_data.get('last_action_result','')[:60]}")

            if done:
                break

        # Grade the episode
        try:
            grade_result = env_grade()
            score = float(grade_result.get("score", 0.0))
        except Exception:
            score = sum(rewards)
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[DEBUG] FaultLine inference starting. Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] Environment URL: {ENV_BASE_URL}", flush=True)

    all_scores = {}
    for task_id in TASKS:
        score = run_task(client, task_id)
        all_scores[task_id] = score
        print(f"[DEBUG] {task_id} final score: {score:.3f}", flush=True)

    print("\n[DEBUG] === FINAL SCORES ===", flush=True)
    for task_id, score in all_scores.items():
        print(f"[DEBUG] {task_id}: {score:.3f}", flush=True)

if __name__ == "__main__":
    main()
