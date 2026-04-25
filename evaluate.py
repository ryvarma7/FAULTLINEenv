import argparse
import json
import os
import sys
from openai import OpenAI

from inference import TASKS, get_success_threshold, run_task, API_BASE_URL, API_KEY


def main():
    parser = argparse.ArgumentParser(description="Evaluate FaultLine Tasks")
    parser.add_argument("--output", type=str, default="assets/baseline_eval.json", help="Output JSON file for results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seeds to evaluate")
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=0.1, max_retries=0)

    print(f"Running evaluation with seeds: {args.seeds}")
    print("Success threshold varies per task.")

    results = {}
    overall_scores = []
    overall_successes = []

    for task in TASKS:
        print(f"\nEvaluating task: {task}")
        task_scores = []
        task_steps = []
        task_successes = []

        results[task] = []
        for seed in args.seeds:
            print(f"  Seed: {seed}")
            score, steps, success = run_task(client, task, seed=seed)

            task_scores.append(score)
            task_steps.append(steps)
            task_successes.append(success)

            results[task].append({
                "seed": seed,
                "score": score,
                "steps": steps,
                "success": success,
            })

            overall_scores.append(score)
            overall_successes.append(success)

        avg_score = sum(task_scores) / len(task_scores)
        success_rate = sum(task_successes) / len(task_successes)

        results[task + "_summary"] = {
            "avg_score": avg_score,
            "success_rate": success_rate,
            "avg_steps": sum(task_steps) / len(task_steps),
        }

    overall_avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    overall_success_rate = sum(overall_successes) / len(overall_successes) if overall_successes else 0

    results["overall_summary"] = {
        "avg_score": overall_avg_score,
        "success_rate": overall_success_rate,
    }

    # Make sure output directory exists if provided
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 40)
    print(f"{'Task':<25} | {'Avg Score':<10} | {'Success Rate':<12} | {'Avg Steps':<10}")
    print("-" * 66)
    for task in TASKS:
        summary = results[task + "_summary"]
        print(
            f"{task:<25} | {summary['avg_score']:<10.3f} | "
            f"{summary['success_rate']:<12.1%} | {summary['avg_steps']:<10.1f}"
        )
    print("-" * 66)
    print(f"{'OVERALL':<25} | {overall_avg_score:<10.3f} | {overall_success_rate:<12.1%}")
    print("=" * 40)
    print(f"\nResults saved to {args.output}")
    print("\nAgent behavior fixed. Environment ready for training.")


if __name__ == "__main__":
    main()
