from faultline.graders.base import BaseGrader, GraderResult
from faultline.tasks.task_medium import TaskMedium

class GraderMedium(BaseGrader):
    """
    Grader for Task 2: Cascading Failure.

    Scoring criteria (total = 1.00):
    - Correct rollback on payment-service:               +0.40
    - Both P1 alerts acknowledged:                       +0.10
    - payment-db logs queried (upstream tracing):        +0.10
    - Postmortem identifies deployment as trigger:       +0.15
    - No wrong rollback taken:                           +0.15
    - Resolved in 12 steps or fewer:                     +0.10
    """

    WEIGHTS = {
        "correct_rollback": 0.40,
        "alerts_acknowledged": 0.10,
        "upstream_logs_queried": 0.10,
        "postmortem_quality": 0.15,
        "no_wrong_actions": 0.15,
        "speed_bonus": 0.10,
    }

    def grade(self, task: TaskMedium) -> GraderResult:
        breakdown: dict = {}
        notes: list = []
        rbd = task.reward_breakdown

        # 1. Correct rollback
        correct = rbd.get("correct_rollback", 0.0) > 0
        breakdown["correct_rollback"] = self.WEIGHTS["correct_rollback"] if correct else 0.0

        # 2. Alerts acknowledged
        ack_count = sum(1 for k in rbd if "ack" in k and rbd[k] > 0)
        breakdown["alerts_acknowledged"] = self.WEIGHTS["alerts_acknowledged"] if ack_count >= 2 else (
            self.WEIGHTS["alerts_acknowledged"] * 0.5 if ack_count == 1 else 0.0)

        # 3. payment-db logs queried
        logs_q = rbd.get("queried_payment_db_logs", 0.0) > 0
        breakdown["upstream_logs_queried"] = self.WEIGHTS["upstream_logs_queried"] if logs_q else 0.0

        # 4. Postmortem quality
        pm_earned = rbd.get("postmortem_quality", 0.0)
        breakdown["postmortem_quality"] = min(self.WEIGHTS["postmortem_quality"], pm_earned)

        # 5. No wrong actions
        wrong = rbd.get("wrong_rollback_target", 0.0) + rbd.get("loop_penalty", 0.0)
        breakdown["no_wrong_actions"] = 0.0 if wrong < -0.05 else self.WEIGHTS["no_wrong_actions"]

        # 6. Speed bonus
        speed = rbd.get("speed_bonus", 0.0) > 0
        breakdown["speed_bonus"] = self.WEIGHTS["speed_bonus"] if speed else 0.0

        score = self._clamp(sum(breakdown.values()))
        return GraderResult(
            score=score,
            breakdown=breakdown,
            passed=score >= 0.40,
            notes=notes,
        )
