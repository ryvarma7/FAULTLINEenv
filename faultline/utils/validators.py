"""
faultline/utils/validators.py
================================
Step output contract enforcer.

GUARANTEE:
  validate_step_output(result) raises immediately with a clear error if the
  step() contract is violated. Use this in server layers to catch env bugs
  before they silently corrupt training trajectories.

Contract:
  Every step() result MUST contain:
    - "observation" : any non-None value
    - "reward"      : int or float
    - "done"        : bool

Usage:
    result = env.step(action)
    validate_step_output(result)   # raises on contract violation
    return result
"""
from __future__ import annotations

from typing import Any, Dict


def validate_step_output(result: Dict[str, Any]) -> bool:
    """
    Validate that a step() result satisfies the OpenEnv contract.

    Args:
        result: The dict returned by env.step().

    Returns:
        True if all checks pass.

    Raises:
        TypeError:  If result is not a dict.
        ValueError: If a required key is missing, or observation is None.
        TypeError:  If reward is not numeric, or done is not bool.
    """
    if not isinstance(result, dict):
        raise TypeError(
            f"step() must return a dict, got {type(result).__name__!r}"
        )

    required = ["observation", "reward", "done"]
    for key in required:
        if key not in result:
            raise ValueError(
                f"step() contract violated: missing required key '{key}'. "
                f"Got keys: {list(result.keys())}"
            )

    if result["observation"] is None:
        raise ValueError("step() contract violated: 'observation' must not be None")

    if not isinstance(result["reward"], (int, float)):
        raise TypeError(
            f"step() contract violated: 'reward' must be numeric, "
            f"got {type(result['reward']).__name__!r} = {result['reward']!r}"
        )

    if not isinstance(result["done"], bool):
        raise TypeError(
            f"step() contract violated: 'done' must be bool, "
            f"got {type(result['done']).__name__!r} = {result['done']!r}"
        )

    return True
