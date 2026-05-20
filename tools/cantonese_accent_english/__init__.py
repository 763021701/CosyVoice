"""Cantonese-accented English CMU phoneme rewrite for CosyVoice3 input."""

from .debug import debug_report_to_json, format_debug_report
from .rewrite import (
    rewrite_english_to_cantonese_accent,
    rewrite_english_to_cantonese_accent_detailed,
)
from .rules_common import WordContext
from .types import (
    AccentPreset,
    AccentRewriteOptions,
    ChangedWord,
    RewriteResult,
)

__all__ = [
    "AccentPreset",
    "AccentRewriteOptions",
    "ChangedWord",
    "RewriteResult",
    "WordContext",
    "debug_report_to_json",
    "format_debug_report",
    "rewrite_english_to_cantonese_accent",
    "rewrite_english_to_cantonese_accent_detailed",
]
