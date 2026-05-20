from __future__ import annotations

import random
from dataclasses import dataclass, field

from .formatting import _phone_base

_CONSONANT_BASES = frozenset({
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",
    "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
})

AE_TO_EH_WORDS = frozenset({
    "man", "bad", "back", "that", "thank", "plan", "family", "happy",
    "camera", "apple",
})


@dataclass
class WordContext:
    word: str
    original_token: str = ""


@dataclass
class RuleApplyResult:
    phones: list[str]
    applied_rules: list[str] = field(default_factory=list)


def is_consonant(phone: str) -> bool:
    return _phone_base(phone) in _CONSONANT_BASES


def maybe_apply(
    rng: random.Random,
    probability: float,
    rule_id: str,
    disabled: set[str],
    enabled: set[str] | None,
) -> bool:
    if rule_id in disabled:
        return False
    if enabled is not None and rule_id not in enabled:
        return False
    if probability >= 1.0:
        return True
    return rng.random() < probability


def weighted_choice(
    rng: random.Random,
    weights: dict[str, float],
) -> str:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    pick = rng.random() * total
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += weight
        if pick < cumulative:
            return key
    return next(reversed(weights))


def replace_all(phones: list[str], src: str, dst: str) -> list[str]:
    return [dst if p == src else p for p in phones]


def replace_phone_at(
    phones: list[str],
    index: int,
    replacements: list[str],
) -> list[str]:
    return phones[:index] + replacements + phones[index + 1:]


def replace_token_globally(
    phones: list[str],
    src: str,
    replacements: list[str],
) -> list[str]:
    out: list[str] = []
    for phone in phones:
        if phone == src:
            out.extend(replacements)
        else:
            out.append(phone)
    return out


def apply_ae_to_eh(
    phones: list[str],
    ctx: WordContext,
    *,
    rng: random.Random,
    probability: float,
    rule_id: str,
    disabled: set[str],
    enabled: set[str] | None,
) -> tuple[list[str], bool]:
    if ctx.word.lower() not in AE_TO_EH_WORDS:
        return phones, False
    if not maybe_apply(rng, probability, rule_id, disabled, enabled):
        return phones, False
    mapping = {"AE": "EH", "AE0": "EH0", "AE1": "EH1", "AE2": "EH2"}
    new = [mapping.get(p, p) for p in phones]
    return new, new != phones


def final_consonant_cluster(phones: list[str]) -> list[int]:
    """Indices of trailing consonants."""
    indices: list[int] = []
    for i in range(len(phones) - 1, -1, -1):
        if is_consonant(phones[i]):
            indices.insert(0, i)
        else:
            break
    return indices


def simplify_final_cluster_td(
    phones: list[str],
    *,
    probability: float,
    rule_id: str,
    rng: random.Random,
    disabled: set[str],
    enabled: set[str] | None,
    applied: list[str],
) -> list[str]:
    if len(phones) < 2:
        return phones
    if phones[-1] not in ("T", "D"):
        return phones
    if not is_consonant(phones[-2]):
        return phones
    if not maybe_apply(rng, probability, rule_id, disabled, enabled):
        return phones
    applied.append(rule_id)
    return phones[:-1]


def vocalize_penultimate_l_before_td(
    phones: list[str],
    *,
    ow0_weight: float,
    delete_weight: float,
    rng: random.Random,
    rule_id: str,
    disabled: set[str],
    enabled: set[str] | None,
    applied: list[str],
) -> list[str]:
    """For patterns like world (W ER1 L D): vocalize L before deleting D."""
    if len(phones) < 3:
        return phones
    if phones[-1] not in ("T", "D") or phones[-2] != "L":
        return phones
    if not is_consonant(phones[-3]):
        return phones
    if not maybe_apply(rng, 1.0, rule_id, disabled, enabled):
        return phones
    choice = weighted_choice(rng, {"OW0": ow0_weight, "DELETE": delete_weight})
    if choice == "DELETE":
        applied.append(f"{rule_id}_DELETE")
        return phones[:-2] + phones[-1:]
    applied.append(f"{rule_id}_OW0")
    # Drop the following T/D coda (e.g. world: W ER1 L D -> W ER1 OW0).
    return phones[:-2] + ["OW0"]


def simplify_long_final_cluster(
    phones: list[str],
    *,
    probability: float,
    rule_id: str,
    rng: random.Random,
    disabled: set[str],
    enabled: set[str] | None,
    applied: list[str],
) -> list[str]:
    """Reduce final consonant clusters of 3+ to at most 2 consonants."""
    cluster_idx = final_consonant_cluster(phones)
    if len(cluster_idx) < 3:
        return phones
    if not maybe_apply(rng, probability, rule_id, disabled, enabled):
        return phones
    applied.append(rule_id)
    keep_from = cluster_idx[-2]
    return phones[:keep_from] + phones[cluster_idx[-2]:]
