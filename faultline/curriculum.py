import collections
import random
from typing import Dict, Any

class CurriculumScheduler:
    STAGES = [
        {
            "cascade_depth": 0,
            "red_herring_count": 0,
            "noise_level": 0.0,
            "multi_region": False,
        },
        {
            "cascade_depth": 1,
            "red_herring_count": 1,
            "noise_level": 0.3,
            "multi_region": False,
        },
        {
            "cascade_depth": 2,
            "red_herring_count": 2,
            "noise_level": 0.6,
            "multi_region": True,
        },
        {
            "cascade_depth": 3,
            "red_herring_count": 3,
            "noise_level": 0.9,
            "multi_region": True,
        }
    ]
    
    THRESHOLDS = [0.5, 0.7, 0.85]
    WINDOW = 30

    def __init__(self):
        self.current_stage = 0
        self.rewards = collections.deque(maxlen=self.WINDOW)

    def record_reward(self, reward: float) -> bool:
        """
        Record a reward and advance the curriculum stage if the average
        reward over the WINDOW meets the current threshold.
        Returns True if the stage advanced.
        """
        self.rewards.append(reward)
        if len(self.rewards) == self.WINDOW:
            avg_reward = sum(self.rewards) / self.WINDOW
            if self.current_stage < len(self.THRESHOLDS) and avg_reward >= self.THRESHOLDS[self.current_stage]:
                self.current_stage += 1
                self.rewards.clear()
                return True
        return False

    def current_config(self) -> Dict[str, Any]:
        """Return the IncidentConfig dictionary for the current stage."""
        config = self.STAGES[self.current_stage].copy()
        
        # Randomly select a failure mode
        failure_modes = [
            "latency",
            "crash",
            "oom",
            "config_drift",
            "connection_leak",
            "quota_exceeded"
        ]
        config["failure_mode"] = random.choice(failure_modes)
        return config
