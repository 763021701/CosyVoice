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
)


def apply_light_rules(
    phones: list[str],
    ctx: WordContext,
    *,
    rng: random.Random,
    disabled_rules: set[str],
    enabled_rules: set[str] | None,
) -> RuleApplyResult:
    result = list(phones)
    applied: list[str] = []

    # 1. TH/DH
    if maybe_apply(rng, 1.0, "TH_TO_F", disabled_rules, enabled_rules):
        new = replace_all(result, "TH", "F")
        if new != result:
            applied.append("TH_TO_F")
            result = new

    if maybe_apply(rng, 1.0, "DH_TO_D", disabled_rules, enabled_rules):
        new = replace_all(result, "DH", "D")
        if new != result:
            applied.append("DH_TO_D")
            result = new

    # 2. V/R/SH (light: no CH/JH)
    if result and result[0] == "V":
        if maybe_apply(rng, 0.6, "V_TO_W", disabled_rules, enabled_rules):
            result[0] = "W"
            applied.append("V_TO_W")

    if result and result[0] == "R":
        if maybe_apply(rng, 0.2, "R_TO_W_LOW", disabled_rules, enabled_rules):
            result[0] = "W"
            applied.append("R_TO_W_LOW")

    if maybe_apply(rng, 0.1, "SH_TO_S_LOW", disabled_rules, enabled_rules):
        new = replace_all(result, "SH", "S")
        if new != result:
            applied.append("SH_TO_S_LOW")
            result = new

    # 3. Vowel substitutions (AE -> EH selected words)
    result, changed = apply_ae_to_eh(
        result,
        ctx,
        rng=rng,
        probability=0.5,
        rule_id="AE_TO_EH_SELECTED_WORDS",
        disabled=disabled_rules,
        enabled=enabled_rules,
    )
    if changed:
        applied.append("AE_TO_EH_SELECTED_WORDS")

    # 4. N/L confusion (very low)
    if result:
        first = result[0]
        if first == "N" and maybe_apply(
            rng, 0.05, "N_L_CONFUSION_VERY_LOW", disabled_rules, enabled_rules,
        ):
            result[0] = "L"
            applied.append("N_L_CONFUSION_VERY_LOW")
        elif first == "L" and maybe_apply(
            rng, 0.05, "N_L_CONFUSION_VERY_LOW", disabled_rules, enabled_rules,
        ):
            result[0] = "N"
            applied.append("N_L_CONFUSION_VERY_LOW")

    # 5. Final L vocalization
    if result and result[-1] == "L":
        if maybe_apply(rng, 0.5, "FINAL_L_VOCALIZATION", disabled_rules, enabled_rules):
            result[-1] = "OW0"
            applied.append("FINAL_L_VOCALIZATION")

    # 6. Final cluster T/D deletion
    before = list(result)
    result = simplify_final_cluster_td(
        result,
        probability=0.7,
        rule_id="FINAL_CLUSTER_T_D_DELETION",
        rng=rng,
        disabled=disabled_rules,
        enabled=enabled_rules,
        applied=applied,
    )
    if result != before and "FINAL_CLUSTER_T_D_DELETION" not in applied:
        pass  # simplify_final_cluster_td appends rule_id

    return RuleApplyResult(phones=result, applied_rules=applied)
