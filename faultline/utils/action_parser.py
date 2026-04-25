"""
faultline/utils/action_parser.py
==================================
Centralized, single-source-of-truth action parser for all layers:
  - faultline/server.py
  - server/app.py
  - Any future inference or evaluation layer

WHY THIS EXISTS:
  Pydantic v2 discriminated unions (FaultLineAction) CANNOT be constructed
  via FaultLineAction(**data) because:
    1. FaultLineAction is a type alias, not a class.
    2. Each concrete subclass uses extra='forbid', which rejects the 'type'
       discriminator key when passed as a kwarg.
  The ONLY correct approach is TypeAdapter(FaultLineAction).validate_python(data).

GUARANTEE:
  parse_action() NEVER raises. All errors are returned as structured dicts.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from pydantic import TypeAdapter, ValidationError

from faultline.models import FaultLineAction

# Module-level singleton — avoids re-creating TypeAdapter on every request.
# TypeAdapter construction is expensive; share it across all callers.
_ACTION_ADAPTER: TypeAdapter[FaultLineAction] = TypeAdapter(FaultLineAction)


def parse_action(
    data: Dict[str, Any]
) -> Tuple[Optional[FaultLineAction], Optional[Dict[str, Any]]]:
    """
    Parse a raw action dictionary into a validated FaultLineAction.

    Returns:
        (action, None)   — on success
        (None, error)    — on any failure; error is a structured dict:
                           {"error": "ACTION_PARSE_ERROR", "message": str(e)}

    NEVER raises. All exceptions are caught and returned as structured errors.

    Usage:
        action, error = parse_action(req.action)
        if error:
            return {"observation": {}, "reward": 0.0, "done": False, **error}
    """
    try:
        action = _ACTION_ADAPTER.validate_python(data)
        return action, None
    except ValidationError as e:
        return None, {
            "error": "ACTION_PARSE_ERROR",
            "error_type": "ValidationError",
            "message": str(e),
        }
    except Exception as e:
        return None, {
            "error": "ACTION_PARSE_ERROR",
            "error_type": type(e).__name__,
            "message": str(e),
        }
