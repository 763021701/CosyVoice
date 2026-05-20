import random

from tools.cantonese_accent_english.rules_common import WordContext
from tools.cantonese_accent_english.rules_strong import apply_strong_rules


def test_ch_to_ts_expansion():
    rng = random.Random(1)  # first draw < 0.5 for CH_TO_TS
    result = apply_strong_rules(
        ["CH", "AY1", "N", "AH0"],
        WordContext(word="china"),
        rng=rng,
        disabled_rules=set(),
        enabled_rules={"CH_TO_TS"},
    )
    assert result.phones == ["T", "S", "AY1", "N", "AH0"]
    assert "CH_TO_TS" in result.applied_rules


def test_jh_to_dz_expansion():
    rng = random.Random(1)  # first draw < 0.5 for JH_TO_DZ
    result = apply_strong_rules(
        ["JH", "AA1", "B"],
        WordContext(word="job"),
        rng=rng,
        disabled_rules=set(),
        enabled_rules={"JH_TO_DZ"},
    )
    assert result.phones == ["D", "Z", "AA1", "B"]


def test_world_penultimate_l_before_td():
    rng = random.Random(0)
    result = apply_strong_rules(
        ["W", "ER1", "L", "D"],
        WordContext(word="world"),
        rng=rng,
        disabled_rules=set(),
        enabled_rules={
            "FINAL_L_VOCALIZATION_OR_DELETION",
            "FINAL_CLUSTER_SIMPLIFICATION",
        },
    )
    assert result.phones == ["W", "ER1", "OW0"]
    assert any("FINAL_L" in r for r in result.applied_rules)


def test_asked_long_cluster_simplification():
    rng = random.Random(0)
    result = apply_strong_rules(
        ["AE1", "S", "K", "T"],
        WordContext(word="asked"),
        rng=rng,
        disabled_rules=set(),
        enabled_rules={"FINAL_CLUSTER_SIMPLIFICATION"},
    )
    assert len(result.phones) <= 4
    assert result.phones[-1] in ("K", "T", "S")
