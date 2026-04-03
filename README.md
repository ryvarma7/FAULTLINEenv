---
title: FaultLine
emoji: 🔥
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
> SRE Incident Response Agent Environment — OpenEnv Hackathon Submission

**"Where agents learn to hold the line."**

## Overview
FaultLine is a fully synthetic, deterministic OpenEnv environment that simulates a production microservices system experiencing a live incident. AI agents navigate multi-step investigations — querying logs, checking metrics, acknowledging alerts — to identify root causes and perform the correct remediations.

**Domain:** Site Reliability Engineering (SRE) — real-time production incident response  
**Why it matters:** Production incidents cost an average of $5,600/minute. No standardized AI benchmark exists for SRE incident response. FaultLine fills that gap.

## Quick Start

### Docker (recommended)
```bash
docker build -t faultline .
docker run -p 7860:7860 faultline
```

### Local
```bash
pip install -r requirements.txt
uvicorn faultline.server:app --host 0.0.0.0 --port 7860
```

### Run baseline inference
```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

## Action Space

| Action | Parameters | Effect |
|--------|-----------|--------|
| `acknowledge_alert` | `alert_id` | Mark alert as owned |
| `query_logs` | `service`, `time_range`, `filter_expr` | Retrieve log entries |
| `check_metrics` | `service`, `metric_name`, `window_minutes` | Get time-series data |
| `rollback` | `service`, `target_version` | Revert a deployment (terminal) |
| `scale_service` | `service`, `replicas` | Increase pod count (terminal) |
| `escalate` | `team`, `message` | Page human team (terminal) |
| `resolve` | `root_cause_service`, `postmortem_text` | Submit root cause analysis (terminal) |

**Available metrics:** `cpu`, `memory`, `latency_p99`, `error_rate`, `throughput`

## Observation Space

Each step returns a `FaultLineObservation` with:
- `alerts`: List of firing alerts with severity (P1–P4), service, title, description
- `log_results`: Log entries from the last `query_logs` action
- `metric_results`: Time-series data from the last `check_metrics` action
- `dependency_graph`: Full service dependency map (12 services)
- `acknowledged_alerts`: Alert IDs the agent has acknowledged
- `elapsed_steps`: Number of actions taken
- `last_action_result`: Human-readable result of the previous action

## Tasks

### Task 1 — Single-Service Latency Spike (Easy)
**Scenario:** `search-service` is alerting HIGH_LATENCY. Root cause is `elasticsearch` running out of heap memory and garbage-collecting aggressively. One alert fires; no red herrings.  
**Key challenge:** Blame the dependency, not the alerting service.  
**Correct terminal action:** `resolve(root_cause_service='elasticsearch', postmortem_text=...)`

### Task 2 — Cascading Failure (Medium)
**Scenario:** A bad `payment-service` deployment introduced a DB connection pool bug. Three alerts fire simultaneously across payment-service, order-service, and payment-db. Agent must trace upstream.  
**Key challenge:** Distinguish the root cause (payment-service deployment) from victims (payment-db, order-service).  
**Correct terminal action:** `rollback(service='payment-service', target_version='v1.4.1')`

### Task 3 — Multi-Region Incident with Red Herrings (Hard)
**Scenario:** `model-serving` hits a CPU quota limit post-model-upgrade, causing fraud-detector timeouts and payment-service fallback to allow-all mode. Two unrelated elasticsearch alerts fire simultaneously as red herrings.  
**Key challenge:** Ignore P3 noise, trace the causal chain, choose `scale_service` not `rollback`.  
**Correct terminal action:** `scale_service(service='model-serving', replicas=4)`

## Reward Function

| Event | Reward |
|-------|--------|
| Acknowledge correct P1/P2 alert (first time) | +0.10 |
| Query logs of root cause service | +0.10 |
| Query anomalous metric (per unique metric) | +0.05 |
| Correct terminal action with right service | +0.35–0.45 |
| Postmortem quality (keyword coverage) | up to +0.15 |
| Speed bonus (resolve under step limit) | +0.05–0.10 |
| Wrong service in terminal action | -0.10 to -0.20 |
| Repeated log query (>3× same service/range) | -0.05 |
| Investigating red herring services (Task 3) | -0.05 per service |
| Escalate to human | +0.05 |

## Baseline Scores

| Task | Random Agent | GPT-4o (est.) | Qwen2.5-72B (est.) | Human Expert |
|------|-------------|---------------|---------------------|-------------|
| Task 1 — Easy | 0.05–0.10 | 0.65–0.80 | 0.60–0.75 | 0.90–1.00 |
| Task 2 — Medium | 0.05–0.10 | 0.40–0.60 | 0.35–0.55 | 0.80–0.95 |
| Task 3 — Hard | 0.00–0.05 | 0.20–0.40 | 0.15–0.35 | 0.70–0.90 |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/reset` | Start new episode `{"task_id": "...", "seed": 42}` |
| POST | `/step` | Execute action `{"action": {...}}` |
| POST | `/state` | Get current observation |
| POST | `/grade` | Grade completed episode |
| GET | `/tasks` | List all tasks |

## Project Structure
```
faultline/
├── faultline/
│   ├── env.py          # FaultLineEnv — core OpenEnv interface
│   ├── models.py       # Pydantic models: Observation, Action, Reward
│   ├── server.py       # FastAPI HTTP server
│   ├── tasks/          # Task implementations (easy/medium/hard)
│   ├── graders/        # Deterministic graders (easy/medium/hard)
│   └── data/           # Synthetic log templates, metric profiles, incident seeds
├── tests/              # pytest test suite
├── inference.py        # Baseline agent (mandatory)
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile
└── requirements.txt
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace / API key |
| `API_BASE_URL` | No | HF router | LLM API endpoint |
| `MODEL_NAME` | No | Qwen2.5-72B | Model identifier |
| `FAULTLINE_URL` | No | localhost:7860 | Environment server URL |

## Testing

Run the full test suite:
```bash
python -m pytest faultline/tests/ -v
```

Expected output: **26 tests passed**

## Architecture

### Environment (`FaultLineEnv`)
- Manages task lifecycle: reset() → observe → step() → done
- Maintains episode state: current task, acknowledged alerts, query history
- Deterministic with seed control

### Tasks
- **BaseTask**: Abstract class with shared mechanics (alerts, logs, metrics generation)
- **TaskEasy**: Single-service latency (elasticsearch OOM)
- **TaskMedium**: Cascading failure (payment-service deployment bug)
- **TaskHard**: Multi-region incident with red herrings (model-serving CPU quota)

### Graders
- **GraderEasy**: Scores based on correct root cause, alert acks, speed
- **GraderMedium**: Scores based on correct rollback, alert triage, log investigation
- **GraderHard**: Scores based on correct scale action, red herring avoidance, postmortem quality

### Server
- FastAPI app serving all environment operations over HTTP
- Stateful session management (one environment per session_id)
- Runs on port 7860 (HF Space default)

## Submission Notes

**Mandatory Output Format** (`inference.py`):
```
[START] task=single_service_latency env=faultline model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.10 done=false error=null
[STEP] step=2 action={...} reward=0.50 done=true error=null
[END] success=true steps=2 score=0.800 rewards=0.10,0.50
```

This format is required for hackathon evaluation. All output goes to stdout with mandatory [START], [STEP], [END] tags.

## License
MIT

## Contact
Built for OpenEnv Hackathon 2025 — SRE Incident Response Challenge
