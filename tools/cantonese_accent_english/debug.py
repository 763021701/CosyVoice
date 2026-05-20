from __future__ import annotations

import json

from .types import RewriteResult


def format_debug_report(result: RewriteResult) -> list[dict[str, object]]:
    """Return per-word debug entries aligned with spec section 25."""
    return [
        {
            "word": entry.original,
            "canonical": entry.canonical_phones,
            "rewritten": entry.rewritten_phones,
            "rules": entry.applied_rules,
        }
        for entry in result.changed_words
    ]


def debug_report_to_json(result: RewriteResult, *, indent: int = 2) -> str:
    return json.dumps(format_debug_report(result), indent=indent, ensure_ascii=False)
