from __future__ import annotations

import random

from .rules_common import (
    RuleApplyResult,
    WordContext,
    apply_ae_to_eh,
    maybe_apply,
    replace_all,
    replace_token_globally,
    simplify_final_cluster_td,
    simplify_long_final_cluster,
    vocalize_penultimate_l_before_td,
    weighted_choice,
)


def _replace_weighted_phone(
    phones: list[str],
    src: str,
    weights: dict[str, float],
    rng: random.Random,
) -> list[str]:
    return [weighted_choice(rng, weights) if p == src else p for p in phones]


def apply_strong_rules(
    phones: list[str],
    ctx: WordContext,
    *,
    rng: random.Random,
    disabled_rules: set[str],
    enabled_rules: set[str] | None,
) -> RuleApplyResult:
    result = list(phones)
    applied: list[str] = []

    # TH/DH weighted
    if any(p == "TH" for p in result):
        if maybe_apply(rng, 1.0, "TH_TO_F_OR_T", disabled_rules, enabled_rules):
            new = _replace_weighted_phone(
                result, "TH", {"F": 0.75, "T": 0.15, "S": 0.10}, rng,
            )
            if new != result:
                applied.append("TH_TO_F_OR_T")
                result = new

    if any(p == "DH" for p in result):
        if maybe_apply(rng, 1.0, "DH_TO_D_OR_Z", disabled_rules, enabled_rules):
            new = _replace_weighted_phone(
                result, "DH", {"D": 0.8, "Z": 0.2}, rng,
            )
            if new != result:
                applied.append("DH_TO_D_OR_Z")
                result = new

    # V/R word-initial weighted
    if result and result[0] == "V":
        if maybe_apply(rng, 1.0, "V_TO_W_OR_F", disabled_rules, enabled_rules):
            result[0] = weighted_choice(rng, {"W": 0.7, "F": 0.3})
            applied.append("V_TO_W_OR_F")

    if result and result[0] == "R":
        if maybe_apply(rng, 1.0, "R_TO_W_OR_L", disabled_rules, enabled_rules):
            result[0] = weighted_choice(rng, {"W": 0.6, "L": 0.4})
            applied.append("R_TO_W_OR_L")

    # SH/CH/JH
    if maybe_apply(rng, 0.5, "SH_TO_S", disabled_rules, enabled_rules):
        new = replace_all(result, "SH", "S")
        if new != result:
            applied.append("SH_TO_S")
            result = new

    if maybe_apply(rng, 0.5, "CH_TO_TS", disabled_rules, enabled_rules):
        new = replace_token_globally(result, "CH", ["T", "S"])
        if new != result:
            applied.append("CH_TO_TS")
            result = new

    if maybe_apply(rng, 0.5, "JH_TO_DZ", disabled_rules, enabled_rules):
        new = replace_token_globally(result, "JH", ["D", "Z"])
        if new != result:
            applied.append("JH_TO_DZ")
            result = new

    # AE -> EH
    result, changed = apply_ae_to_eh(
        result,
        ctx,
        rng=rng,
        probability=0.8,
        rule_id="AE_TO_EH_SELECTED_WORDS",
        disabled=disabled_rules,
        enabled=enabled_rules,
    )
    if changed:
        applied.append("AE_TO_EH_SELECTED_WORDS")

    # Final L: penultimate L before T/D (world) — always when pattern matches (spec §12 order)
    if len(result) >= 3 and result[-1] in ("T", "D") and result[-2] == "L":
        if maybe_apply(rng, 1.0, "FINAL_L_VOCALIZATION_OR_DELETION", disabled_rules, enabled_rules):
            before = list(result)
            result = vocalize_penultimate_l_before_td(
                result,
                ow0_weight=0.7,
                delete_weight=0.3,
                rng=rng,
                rule_id="FINAL_L_VOCALIZATION_OR_DELETION",
                disabled=disabled_rules,
                enabled=enabled_rules,
                applied=applied,
            )
            if result == before:
                result = result[:-2] + ["OW0"]
                applied.append("FINAL_L_VOCALIZATION_OR_DELETION_OW0")

    # Word-final L only (probability 0.8)
    if result and result[-1] == "L":
        if maybe_apply(rng, 0.8, "FINAL_L_VOCALIZATION_OR_DELETION", disabled_rules, enabled_rules):
            choice = weighted_choice(rng, {"OW0": 0.7, "DELETE": 0.3})
            if choice == "DELETE":
                result = result[:-1]
                applied.append("FINAL_L_VOCALIZATION_OR_DELETION_DELETE")
            else:
                result[-1] = "OW0"
                applied.append("FINAL_L_VOCALIZATION_OR_DELETION_OW0")

    # Final cluster simplification
    before = list(result)
    result = simplify_final_cluster_td(
        result,
        probability=0.9,
        rule_id="FINAL_CLUSTER_SIMPLIFICATION",
        rng=rng,
        disabled=disabled_rules,
        enabled=enabled_rules,
        applied=applied,
    )
    result = simplify_long_final_cluster(
        result,
        probability=0.9,
        rule_id="FINAL_CLUSTER_SIMPLIFICATION",
        rng=rng,
        disabled=disabled_rules,
        enabled=enabled_rules,
        applied=applied,
    )

    # Vowel insertion after final S or KS
    if maybe_apply(rng, 0.3, "FINAL_S_KS_VOWEL_INSERTION", disabled_rules, enabled_rules):
        if len(result) >= 2 and result[-2] == "K" and result[-1] == "S":
            result = result + ["IY0"]
            applied.append("FINAL_S_KS_VOWEL_INSERTION")
        elif result and result[-1] == "S":
            result = result + ["IY0"]
            applied.append("FINAL_S_KS_VOWEL_INSERTION")

    return RuleApplyResult(phones=result, applied_rules=applied)
