"""CMU pronunciation lookup via CMUdict (all standard English words)."""

from __future__ import annotations

from functools import lru_cache

import pronouncing

from .formatting import VALID_CMU_TOKENS, has_vowel

# Optional manual fixes (British spellings, domain terms) checked before CMUdict.
_MANUAL_OVERRIDES: dict[str, list[str]] = {
    # Keep empty by default; add entries here only when CMUdict lacks a form.
}

# Common UK -> US spellings so CMUdict can resolve both.
_SPELLING_VARIANTS: dict[str, str] = {
    "labelled": "labeled",
    "coloured": "colored",
    "favour": "favor",
    "favourite": "favorite",
    "centre": "center",
    "metre": "meter",
    "organise": "organize",
    "organised": "organized",
}


def _parse_pronouncing(phone_str: str) -> list[str]:
    return phone_str.split()


def _normalize_phones(phones: list[str]) -> list[str] | None:
    if not phones:
        return None
    if not all(p in VALID_CMU_TOKENS for p in phones):
        return None
    if not has_vowel(phones):
        return None
    return phones


def _lookup_pronouncing(word: str) -> list[str] | None:
    candidates = pronouncing.phones_for_word(word)
    if not candidates:
        return None
    for candidate in candidates:
        phones = _normalize_phones(_parse_pronouncing(candidate))
        if phones is not None:
            return phones
    return None


@lru_cache(maxsize=65536)
def lookup_cmu(word: str) -> list[str] | None:
    """Return CMU/ARPAbet phones for *word*, or None if unavailable."""
    key = word.lower().replace("'", "")

    if key in _MANUAL_OVERRIDES:
        phones = _normalize_phones(list(_MANUAL_OVERRIDES[key]))
        if phones is not None:
            return phones

    phones = _lookup_pronouncing(key)
    if phones is not None:
        return phones

    variant = _SPELLING_VARIANTS.get(key)
    if variant is not None:
        phones = _lookup_pronouncing(variant)
        if phones is not None:
            return phones

    return None
