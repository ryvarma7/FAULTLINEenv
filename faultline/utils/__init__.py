"""
faultline/utils — centralized utility package.

Exports:
  parse_action(data)           → (action, None) | (None, error_dict)
  validate_step_output(result) → True | raises ValueError/TypeError
"""
from faultline.utils.action_parser import parse_action
from faultline.utils.validators import validate_step_output

__all__ = ["parse_action", "validate_step_output"]
