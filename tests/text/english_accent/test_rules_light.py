import random

from tools.cantonese_accent_english.rules_common import WordContext
from tools.cantonese_accent_english.rules_light import apply_light_rules


def test_ae_to_eh_selected_word():
    rng = random.Random(1)  # first draw < 0.5 for AE rule
    result = apply_light_rules(
        ["M", "AE1", "N"],
        WordContext(word="man"),
        rng=rng,
        disabled_rules=set(),
        enabled_rules={"AE_TO_EH_SELECTED_WORDS"},
    )
    assert result.phones == ["M", "EH1", "N"]
    assert "AE_TO_EH_SELECTED_WORDS" in result.applied_rules


def test_ae_not_applied_outside_wordlist():
    rng = random.Random(0)
    result = apply_light_rules(
        ["K", "AE1", "T"],
        WordContext(word="cat"),
        rng=rng,
        disabled_rules=set(),
        enabled_rules={"AE_TO_EH_SELECTED_WORDS"},
    )
    assert result.phones == ["K", "AE1", "T"]


def test_no_ch_to_ts_in_light():
    rng = random.Random(0)
    result = apply_light_rules(
        ["CH", "AY1", "N", "AH0"],
        WordContext(word="china"),
        rng=rng,
        disabled_rules=set(),
        enabled_rules=None,
    )
    assert "CH" in result.phones
    assert "T" not in result.phones or result.phones.count("T") == 0
