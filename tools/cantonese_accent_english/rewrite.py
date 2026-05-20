from __future__ import annotations

import random

from .cmudict import lookup_cmu
from .formatting import (
    TextToken,
    detokenize,
    format_cmu_phones,
    tokenize_text,
    validate_phones,
)
from .presets_overrides import get_default_overrides
from .rules_common import WordContext
from .rules_light import apply_light_rules
from .rules_strong import apply_strong_rules
from .types import AccentRewriteOptions, ChangedWord, RewriteResult

_SENTENCE_END = frozenset(".?!")

_DEFAULT_MAX_REWRITES = {"light": 8, "strong": 15}

_COMMON_SENTENCE_STARTERS = frozenset({
    "a", "an", "and", "as", "at", "but", "for", "from", "he", "her", "his",
    "i", "if", "in", "is", "it", "my", "no", "not", "of", "on", "or", "our",
    "she", "so", "the", "their", "them", "then", "there", "they", "this",
    "to", "up", "us", "we", "what", "when", "where", "who", "with", "you",
    "your",
})


def _create_rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def _is_acronym(token: str) -> bool:
    letters = [c for c in token if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def _should_preserve_proper_noun(token: TextToken) -> bool:
    if not token.is_word:
        return False
    word = token.text
    if word.islower():
        return False
    if token.is_sentence_end and word.lower() in _COMMON_SENTENCE_STARTERS:
        return False
    if token.is_sentence_end:
        return False
    return word[0].isupper()


def _should_skip_word(token: TextToken, options: AccentRewriteOptions) -> bool:
    if options.preserve_acronyms and _is_acronym(token.text):
        return True
    if options.preserve_proper_nouns and _should_preserve_proper_noun(token):
        return True
    return False


def _normalize_word_key(token: str) -> str:
    return token.lower().replace("'", "")


def _resolve_overrides(options: AccentRewriteOptions) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    if options.use_default_overrides:
        merged.update(get_default_overrides(options.preset))
    for word, phones in options.custom_word_overrides.items():
        merged[word.lower()] = list(phones)
    return merged


def _effective_max_rewrites(options: AccentRewriteOptions) -> int | None:
    if options.max_rewrites_per_sentence is not None:
        return options.max_rewrites_per_sentence
    return _DEFAULT_MAX_REWRITES.get(options.preset)


def _apply_preset_rules(
    canonical: list[str],
    ctx: WordContext,
    options: AccentRewriteOptions,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    if options.preset == "light":
        result = apply_light_rules(
            canonical,
            ctx,
            rng=rng,
            disabled_rules=options.disabled_rules,
            enabled_rules=options.enabled_rules,
        )
    elif options.preset == "strong":
        result = apply_strong_rules(
            canonical,
            ctx,
            rng=rng,
            disabled_rules=options.disabled_rules,
            enabled_rules=options.enabled_rules,
        )
    else:
        raise ValueError(f"Unknown preset: {options.preset!r}")
    return result.phones, result.applied_rules


def rewrite_english_to_cantonese_accent_detailed(
    text: str,
    options: AccentRewriteOptions,
) -> RewriteResult:
    if options.preset not in ("light", "strong"):
        raise ValueError(f"Unknown preset: {options.preset!r}")

    rng = _create_rng(options.seed)
    overrides = _resolve_overrides(options)
    max_rewrites = _effective_max_rewrites(options)
    tokens = tokenize_text(text)
    output_pieces: list[str] = []
    changed_words: list[ChangedWord] = []
    rewrite_count = 0
    char_offset = 0

    for token in tokens:
        if not token.is_word:
            output_pieces.append(token.text)
            if token.text in _SENTENCE_END:
                rewrite_count = 0
            char_offset += len(token.text)
            continue

        original = token.text
        lower = _normalize_word_key(original)

        if _should_skip_word(token, options):
            output_pieces.append(original)
            char_offset += len(original)
            continue

        if max_rewrites is not None and rewrite_count >= max_rewrites:
            output_pieces.append(original)
            char_offset += len(original)
            continue

        canonical = lookup_cmu(lower)
        if canonical is None:
            output_pieces.append(original)
            char_offset += len(original)
            continue

        applied_rules: list[str] = []
        ctx = WordContext(word=lower, original_token=original)

        if lower in overrides:
            rewritten = list(overrides[lower])
            applied_rules.append("CUSTOM_WORD_OVERRIDE")
        else:
            rewritten, applied_rules = _apply_preset_rules(
                canonical, ctx, options, rng,
            )

        if not validate_phones(rewritten):
            output_pieces.append(original)
            char_offset += len(original)
            continue

        changed = canonical != rewritten
        if changed or options.replace_all_words:
            piece = format_cmu_phones(rewritten)
            output_pieces.append(piece)
            if changed:
                rewrite_count += 1
                changed_words.append(
                    ChangedWord(
                        original=original,
                        canonical_phones=canonical,
                        rewritten_phones=rewritten,
                        applied_rules=applied_rules,
                        start_index=char_offset,
                        end_index=char_offset + len(piece),
                    )
                )
        else:
            output_pieces.append(original)

        char_offset += len(output_pieces[-1])

    return RewriteResult(
        original_text=text,
        rewritten_text=detokenize(output_pieces),
        preset=options.preset,
        changed_words=changed_words,
    )


def rewrite_english_to_cantonese_accent(
    text: str,
    options: AccentRewriteOptions,
) -> str:
    return rewrite_english_to_cantonese_accent_detailed(text, options).rewritten_text
