from faultline.graders.base import BaseGrader, GraderResult
from faultline.tasks.task_easy import TaskEasy

class GraderEasy(BaseGrader):
    """
    Grader for Task 1: Single-Service Latency Spike.

    Scoring criteria (total = 1.00):
    - Correct root cause service named in resolve():     +0.45
    - P2 alert acknowledged:                             +0.10
    - elasticsearch logs queried:                        +0.10
    - Postmortem mentions GC or memory keyword:          +0.15
    - Resolved in 6 steps or fewer:                      +0.10
    - No wrong rollback actions taken:                   +0.10
    """

    WEIGHTS = {
        "correct_root_cause": 0.45,
        "alert_acknowledged": 0.10,
        "root_cause_logs_queried": 0.10,
        "postmortem_quality": 0.15,
        "speed_bonus": 0.10,
        "no_wrong_actions": 0.10,
    }

    def grade(self, task: TaskEasy) -> GraderResult:
        breakdown: dict = {}
        notes: list = []

        # 1. Correct root cause
        rbd = task.reward_breakdown
        correct_root_cause = rbd.get("correct_root_cause", 0.0) > 0
        breakdown["correct_root_cause"] = self.WEIGHTS["correct_root_cause"] if correct_root_cause else 0.0
        if not correct_root_cause:
            notes.append("Root cause service was incorrect or resolve() not called.")

        # 2. Alert acknowledged
        alerted = any("ack" in k for k in rbd)
        breakdown["alert_acknowledged"] = self.WEIGHTS["alert_acknowledged"] if alerted else 0.0

        # 3. Root cause logs queried
        logs_queried = rbd.get("queried_root_cause_logs", 0.0) > 0
        breakdown["root_cause_logs_queried"] = self.WEIGHTS["root_cause_logs_queried"] if logs_queried else 0.0

        # 4. Postmortem quality
        pm_earned = rbd.get("postmortem_quality", 0.0)
        breakdown["postmortem_quality"] = min(self.WEIGHTS["postmortem_quality"], pm_earned)

        # 5. Speed bonus
        speed = rbd.get("speed_bonus", 0.0) > 0
        breakdown["speed_bonus"] = self.WEIGHTS["speed_bonus"] if speed else 0.0

        # 6. No wrong actions
        wrong = rbd.get("wrong_root_cause", 0.0)
        breakdown["no_wrong_actions"] = 0.0 if wrong < 0 else self.WEIGHTS["no_wrong_actions"]

        score = self._clamp(sum(breakdown.values()))
        return GraderResult(
            score=score,
            breakdown=breakdown,
            passed=score >= 0.45,
            notes=notes,
        )
