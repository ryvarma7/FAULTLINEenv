from faultline.graders.base import BaseGrader, GraderResult
from faultline.tasks.task_hard import TaskHard

class GraderHard(BaseGrader):
    """
    Grader for Task 3: Multi-Region Incident with Red Herrings.

    Scoring criteria (total = 1.00):
    - Correct root cause (model-serving):                +0.35
    - Correct action type (scale_service not rollback):  +0.15
    - P1 alerts ack'd, P3 red herrings ignored:         +0.10
    - fraud-detector and model-serving logs queried:     +0.10
    - Postmortem identifies quota limit / model upgrade: +0.15
    - No wasted investigation on red herrings:           +0.10
    - Resolved in 16 steps or fewer:                     +0.05
    """

    WEIGHTS = {
        "correct_root_cause": 0.35,
        "correct_action_type": 0.15,
        "alert_triage": 0.10,
        "correct_logs_queried": 0.10,
        "postmortem_quality": 0.15,
        "no_red_herring_waste": 0.10,
        "speed_bonus": 0.05,
    }

    def grade(self, task: TaskHard) -> GraderResult:
        breakdown: dict = {}
        notes: list = []
        rbd = task.reward_breakdown

        # 1. Correct root cause (model-serving scaled successfully)
        correct = rbd.get("correct_scale_service", 0.0) > 0
        breakdown["correct_root_cause"] = self.WEIGHTS["correct_root_cause"] if correct else 0.0

        # 2. Correct action type (scale not rollback)
        correct_action = rbd.get("correct_scale_service", 0.0) > 0
        wrong_type = rbd.get("wrong_action_rollback", 0.0) < 0
        breakdown["correct_action_type"] = self.WEIGHTS["correct_action_type"] if (correct_action and not wrong_type) else 0.0

        # 3. Alert triage (P1 acked, no penalty for P3 acks recorded)
        ack_earned = sum(v for k, v in rbd.items() if "ack" in k and v > 0)
        breakdown["alert_triage"] = min(self.WEIGHTS["alert_triage"], ack_earned)

        # 4. Correct logs queried (fraud-detector and model-serving)
        logs_q = (rbd.get("queried_fraud_detector_logs", 0.0) > 0 or 
                  rbd.get("queried_model_serving_logs", 0.0) > 0)
        breakdown["correct_logs_queried"] = self.WEIGHTS["correct_logs_queried"] if logs_q else 0.0

        # 5. Postmortem quality
        pm_earned = rbd.get("postmortem_quality", 0.0)
        breakdown["postmortem_quality"] = min(self.WEIGHTS["postmortem_quality"], pm_earned)

        # 6. No red herring waste
        red_herring_penalties = sum(v for k, v in rbd.items() if "red_herring" in k and v < 0)
        breakdown["no_red_herring_waste"] = 0.0 if red_herring_penalties < -0.04 else self.WEIGHTS["no_red_herring_waste"]

        # 7. Speed bonus
        speed = rbd.get("speed_bonus", 0.0) > 0
        breakdown["speed_bonus"] = self.WEIGHTS["speed_bonus"] if speed else 0.0

        score = self._clamp(sum(breakdown.values()))
        return GraderResult(
            score=score,
            breakdown=breakdown,
            passed=score >= 0.35,
            notes=notes,
        )
