# FaultLine: Training an AI SRE Agent for Production Incident Response

FaultLine is an OpenEnv environment built to train and evaluate AI agents on production incident response. The system simulates a 12-service microservices architecture under active incidents, requiring agents to navigate partial observability, triage alerts, query logs, check metrics, and execute correct remediation actions in a fully deterministic environment.

## Why the Problem Matters
Production incidents are expensive. Current LLMs consistently fail at structured SRE triage—they hallucinate actions, investigate the wrong services, and miss causal chains between dependencies. Before FaultLine, there was no standardized benchmark or environment to measure or train this specific capability without risking actual production infrastructure.

## Environment Architecture and Agent Interaction
The environment operates via a REST API where the agent receives observations containing firing alerts, log entries, time-series metrics, and the dependency graph. The agent must interact with the system by executing discrete actions: acknowledging alerts, querying logs, checking metrics, or reading runbooks. Once the agent determines the root cause, it must submit a terminal action, such as rolling back a deployment or scaling a service, alongside a structured postmortem. FaultLine includes three hand-crafted incident scenarios and a procedural generator for infinite curriculum training.

## Training Pipeline and Framework
We trained a base model, Qwen2.5-1.5B-Instruct, using a two-stage approach:
1. **Supervised Fine-Tuning (SFT)**: 210 steps on 50 expert trajectories covering over 15 distinct failure modes.
2. **Group Relative Policy Optimization (GRPO)**: 50 steps using the environment's deterministic grading function as the reward signal. 

## Training Evidence
The model successfully learned to navigate the environment and maximize the reward function. SFT rapidly taught the model proper JSON action syntax and basic SRE reasoning patterns. GRPO fine-tuning subsequently increased the average reward score from 2.5/6.0 (baseline SFT) to 5.7/6.0.

<p align="center"><img src="https://github.com/ryvarma7/FAULTLINEenv/blob/main/assets/grpo_before_after.png?raw=true" width="550" alt="GRPO Before and After"></p>

## Results and Improvements
The training pipeline yielded significant, measurable improvements. Between the base model and the final SFT+GRPO agent:
- The average environment score increased from 0.05 to 0.85.
- The rate of invalid actions per episode dropped from 6 down to 0.

<p align="center"><img src="https://github.com/ryvarma7/FAULTLINEenv/blob/main/assets/before_and_after_benchmark.png?raw=true" width="550" alt="Before and After Benchmark"></p>

The environment infrastructure is fully hardened, having passed 22/22 stress tests covering deterministic grading, seed reproducibility, and loop penalty enforcement.

## Project Links
Hugging Face Space: https://huggingface.co/spaces/ryvarma7/faultline
GitHub Repository: https://github.com/ryvarma7/FAULTLINEenv
Training Notebook (Kaggle): https://www.kaggle.com/code/ryvarma7/faultline-sft-grpo
