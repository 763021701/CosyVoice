from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AccentPreset = Literal["light", "strong"]


@dataclass
class AccentRewriteOptions:
    preset: AccentPreset
    seed: int | None = None
    replace_all_words: bool = False
    max_rewrites_per_sentence: int | None = None
    preserve_proper_nouns: bool = True
    preserve_acronyms: bool = True
    use_default_overrides: bool = False
    custom_word_overrides: dict[str, list[str]] = field(default_factory=dict)
    disabled_rules: set[str] = field(default_factory=set)
    enabled_rules: set[str] | None = None


@dataclass
class ChangedWord:
    original: str
    canonical_phones: list[str]
    rewritten_phones: list[str]
    applied_rules: list[str]
    start_index: int | None = None
    end_index: int | None = None


@dataclass
class RewriteResult:
    original_text: str
    rewritten_text: str
    preset: AccentPreset
    changed_words: list[ChangedWord] = field(default_factory=list)
