"""Backward-compatible re-exports; prefer rules_light / rules_strong."""

from .rules_common import RuleApplyResult, WordContext
from .rules_light import apply_light_rules

__all__ = ["RuleApplyResult", "WordContext", "apply_light_rules"]
