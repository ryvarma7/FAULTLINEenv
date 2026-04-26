---
title: FaultLine
emoji: "\U0001F525"
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
tags:
  - openenv
  - sre
  - incident-response
  - agent
license: mit
---

# FaultLine

An OpenEnv environment for training and evaluating AI agents on production incident response.

Agents navigate a simulated 12-service microservices system under active incidents -- triaging alerts, querying logs, checking metrics, and executing the correct remediation. The environment is fully synthetic, deterministic, and seed-reproducible.

**Domain:** Site Reliability Engineering (SRE).

---

## Problem

Production incidents cost thousands per minute. Current LLMs fail at structured SRE triage: they hallucinate actions, investigate wrong services, and miss causal chains between dependencies. No standardized benchmark exists to measure or train this capability.

FaultLine provides that benchmark -- three hand-crafted incident scenarios with deterministic grading, a procedural generator for infinite curriculum training, and a reward function designed to resist gaming.

---

## Hackathon Theme Alignment

**Theme 3 -- World Modeling (3.1 Professional Tasks)**

FaultLine is a partially observable world model of a production microservices system. The agent cannot see the full system state -- it must actively query logs, check metrics, and triage alerts to build a causal picture of what is happening. Each action updates the agent's belief about root cause and narrows the investigation. The environment requires multi-step tool orchestration (not single-shot Q&A), real causal reasoning across a 12-service dependency graph, and correct terminal action selection. This is the kind of professional SRE workflow that cannot be solved by pattern matching or shortcut exploitation.

**Theme 4 -- Self-Improvement**

FaultLine includes a procedural incident generator (`faultline/generator.py`) and a 4-stage curriculum scheduler (`faultline/curriculum.py`). The generator produces infinite unique incident scenarios from 6 failure modes, configurable cascade depth (0-3), red herring count (0-3), and noise levels. The curriculum scheduler auto-advances difficulty when the agent's rolling average reward crosses stage thresholds (0.5, 0.7, 0.85), creating an adaptive training loop where the environment scales with the agent's capability. This is recursive skill amplification -- the agent's performance directly drives the difficulty of its next training batch.

---

## Links

| Resource | URL |
|----------|-----|
| Hugging Face Space | `[TODO: Add HF Space URL]` |
| Training Notebook | `[TODO: Add Kaggle/Colab notebook URL]` |
| Blog / Video | `[TODO: Add blog post or video URL]` |
| GitHub Repo | `[TODO: Add GitHub URL]` |

---

## How the Environment Works

```
Agent                         FaultLine Server (port 7860)
  |                                  |
  |--- POST /reset {task, seed} ---->|   Initialize incident scenario
  |<--- observation (alerts, graph) -|
  |                                  |
  |--- POST /step {action} -------->|   Execute action, return reward
  |<--- {obs, reward, done, info} --|
  |          ... loop ...            |
  |--- POST /grade ---------------->|   Final deterministic score
  |<--- {score, breakdown, passed} -|
```

Each observation includes:
- Firing alerts with severity (P1-P4), service, and description
- Log entries from `query_logs` actions
- Time-series metrics from `check_metrics` actions
- Full 12-service dependency graph
- Acknowledged alert list
- Step count and last action result

### Action Space

| Action | Parameters | Terminal |
|--------|-----------|---------|
| `acknowledge_alert` | `alert_id` | No |
| `query_logs` | `service`, `time_range`, `filter_expr` | No |
| `check_metrics` | `service`, `metric_name`, `window_minutes` | No |
| `query_runbook` | `topic` | No |
| `rollback` | `service`, `target_version` | Yes |
| `scale_service` | `service`, `replicas` | Yes |
| `resolve` | `root_cause_service`, `postmortem_text` | Yes |
| `escalate` | `team`, `message` | No |

Available metrics: `cpu`, `memory`, `latency_p99`, `error_rate`, `throughput`

Max steps per episode: 20

---

## Tasks

### Task 1 -- Single Service Latency (Easy)

`search-service` alerts on high latency. Root cause: `elasticsearch` heap exhaustion causing GC storms. One alert, no red herrings.

Challenge: Blame the dependency, not the alerting service.

Correct action: `resolve(root_cause_service='elasticsearch', postmortem_text=...)`

### Task 2 -- Cascading Failure (Medium)

A bad `payment-service` deployment leaks database connections. Three alerts fire across `payment-service`, `order-service`, and `payment-db`. Agent must trace upstream.

Challenge: Distinguish the root cause deployment from the downstream victims.

Correct action: `rollback(service='payment-service', target_version='v1.4.1')`

### Task 3 -- Multi-Region Incident with Red Herrings (Hard)

`model-serving` hits CPU quota after a model upgrade, causing `fraud-detector` timeouts and `payment-service` fallback. Two unrelated elasticsearch alerts fire as noise.

Challenge: Ignore P3 distractors, trace the causal chain, pick `scale_service` over `rollback`.

Correct action: `scale_service(service='model-serving', replicas=4)`

---

## Reward Structure

| Criterion | Easy | Medium | Hard |
|-----------|------|--------|------|
| Correct Terminal Action / Root Cause | 0.45 | 0.40 | 0.50 |
| Alert Triage | 0.10 | 0.10 | 0.10 |
| Logs/Metrics Investigation | 0.10 | 0.10 | 0.10 |
| Postmortem Quality | 0.15 | 0.15 | 0.15 |
| Speed Bonus | 0.10 | 0.10 | 0.05 |
| No Wrong Actions / Red Herrings | 0.10 | 0.15 | 0.10 |
| **Total** | **1.00** | **1.00** | **1.00** |

Penalties: repeated log queries (>3x same key), runbook overuse (>3 queries), wrong terminal actions, red herring investigation. Postmortems are graded on structural quality and keyword density to prevent reward hacking.

---

## What Was Trained

**Base model:** Qwen2.5-1.5B-Instruct

**Pipeline:**
1. **SFT** -- Supervised fine-tuning on 50 expert trajectories (`quality_data.json`). Each trajectory is a complete incident resolution with correct observation-action pairs and detailed postmortems. 210 training steps.
2. **GRPO** -- Group Relative Policy Optimization using the FaultLine environment reward signal. 50 training steps. The model learns to maximize the environment's grading function directly.

**Training dataset:** 50 curated trajectories covering 15+ distinct failure modes (OOM, connection leak, quota exhaustion, config drift, crash loops, replication lag, deadlocks, TLS expiry, etc.)

---

## Training Evidence

### SFT Loss Curve

The model learns JSON action syntax and SRE reasoning patterns rapidly in the first 50 steps, converging to loss < 0.35 by step 210.

<p align="center"><img src="assets/sft_detailed_loss.png" width="700"></p>

### GRPO Reward Progression

Mean reward over 50 GRPO training steps. The upward trend shows the model learning to select higher-reward action sequences.

<p align="center"><img src="assets/grpo_reward.png" width="700"></p>

### GRPO Loss

Policy loss during GRPO fine-tuning. The spike at step ~34 corresponds to an exploration penalty that the model recovers from.

<p align="center"><img src="assets/grpo_loss.png" width="700"></p>

### GRPO Learning Curve (Annotated)

Raw reward per step with moving average trendline. Annotated with exploration penalty and mastery phases.

<p align="center"><img src="assets/grpo_annotated_learning_curve.png" width="700"></p>

### Before vs. After

Environment score improvement from base model (0.05) to final SFT+GRPO agent (0.85). Invalid actions per episode dropped from 6 to 0.

<p align="center"><img src="assets/before_and_after_benchmark.png" width="550"></p>

### Agent Evolution Across Pipeline Stages

Score and invalid action count at each stage: Initial Model, Broken RL, Stabilized SFT, Final Agent (SFT + GRPO).

<p align="center"><img src="assets/agent_evolution_dual_axis.png" width="700"></p>

### GRPO Before vs. After (Reward Score)

Average reward score from 2.5/6.0 (baseline SFT) to 5.7/6.0 (after GRPO).

<p align="center"><img src="assets/grpo_before_after.png" width="550"></p>

---

## Results

### Stress Test Suite (22/22 passed)

From `stress_test_results.json`:

| Test Category | Result |
|--------------|--------|
| Task initialization (all 3 tasks) | Passed |
| Seed reproducibility (all 3 tasks) | Passed |
| Action execution (ack, logs, metrics) | Passed |
| Episode completion (easy) | 4 steps, done=True |
| Grading: Easy | 0.850, passed |
| Grading: Medium | 0.850, passed |
| Grading: Hard | 0.850, passed |
| Invalid task/state handling | Passed |
| Loop penalty enforcement | Passed |
| Dependency graph validation (12 services) | Passed |
| Stress load (9 episodes, 0 failures) | Passed |
| Observation contract validity | Passed |

### Deterministic Baseline Scores

Expert agent following validated solution paths:

| Task | Score | Correct Action | Speed Bonus | Clean Execution |
|------|-------|---------------|-------------|-----------------|
| Easy | 0.850 | 0.45 / 0.45 | +0.10 | +0.10 |
| Medium | 0.850 | 0.40 / 0.40 | +0.10 | +0.15 |
| Hard | 0.850 | 0.50 (root cause) + 0.15 (action type) | +0.05 | +0.10 |

### Agent Performance (Post-Training)

| Metric | Before (Base Model) | After (SFT + GRPO) |
|--------|--------------------|--------------------|
| Avg environment score | 0.05 | 0.85 |
| Invalid actions per episode | 6 | 0 |
| Avg reward (sum across tasks) | 2.5 / 6.0 | 5.7 / 6.0 |

---

## Reproducibility

### Run the environment

```bash
# Docker
docker build -t faultline .
docker run -p 7860:7860 faultline

# Local
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run inference

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN
python inference.py
```

### Run the test suite

```bash
python -m pytest faultline/tests/ -v
python -m pytest tests/ -v
```

### Run debug sanity check (no server needed)

```bash
python debug_run.py --all
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | -- | HuggingFace API token |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `meta-llama/Llama-3.1-405B-Instruct` | Model identifier |
| `FAULTLINE_URL` | `http://127.0.0.1:7860` | Environment server URL |
| `MAX_STEPS` | `15` | Max inference steps |
| `TEMPERATURE` | `0.2` | LLM temperature |

### Seed control

All tasks are deterministic with seed control. Default seed: `42`. Same seed produces identical episodes, observations, and grading.

---

## Project Structure

```
FAULTLINEenv/
  faultline/
    env.py              Core OpenEnv environment (reset, step, grade)
    models.py            Pydantic v2 models (observations, actions, rewards)
    server.py            Internal server module
    generator.py         Procedural incident generator (infinite scenarios)
    curriculum.py        Curriculum scheduler (4-stage difficulty progression)
    runbooks.py          SRE runbook knowledge base (10 topics)
    tasks/
      base.py            Abstract task class
      task_easy.py       Single-service latency scenario
      task_medium.py     Cascading failure scenario
      task_hard.py       Multi-region incident scenario
    graders/
      base.py            Grader interface + postmortem scorer
      grader_easy.py     Easy task grading logic
      grader_medium.py   Medium task grading logic
      grader_hard.py     Hard task grading logic
    data/
      incidents/         Incident seed data (easy.json, medium.json, hard.json)
      log_templates.json Synthetic log templates
      metric_profiles.json Metric generation profiles
    utils/
      action_parser.py   Centralized Pydantic v2 discriminated union parser
      validators.py      Step output contract validator
  server/
    app.py               FastAPI application (HTTP API layer)
  tests/
    test_contracts.py    Contract and integration tests
  assets/
    sft_detailed_loss.png
    grpo_reward.png
    grpo_loss.png
    grpo_annotated_learning_curve.png
    grpo_before_after.png
    before_and_after_benchmark.png
    agent_evolution_dual_axis.png
    baseline_eval.json
  inference.py           ReAct-style LLM agent (mandatory submission format)
  evaluate.py            Multi-seed evaluation harness
  debug_run.py           Local sanity check (no server needed)
  debug_sanity.py        Additional debug utilities
  quality_data.json      50 expert SRE trajectories (SFT training data)
  stress_test_results.json  22/22 environment stress tests
  openenv.yaml           OpenEnv metadata
  Dockerfile
  requirements.txt
  pyproject.toml
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/reset` | Start episode: `{"task_id": "...", "seed": 42}` |
| POST | `/step` | Execute action: `{"action": {...}}` |
| POST | `/state` | Get current observation |
| POST | `/grade` | Grade completed episode |
| GET | `/tasks` | List available tasks |

---

## Inference Output Format

```
[START] task=single_service_latency env=faultline model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.10 done=false error=null
[STEP] step=2 action={...} reward=0.50 done=true error=null
[END] success=true steps=2 score=0.800 rewards=0.10,0.50
```

Required for hackathon evaluation. All output to stdout with `[START]`, `[STEP]`, `[END]` tags.

---

## License

MIT

Built for OpenEnv Hackathon 2026.
