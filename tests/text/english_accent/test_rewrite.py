import pytest

from tools.cantonese_accent_english import (
    AccentRewriteOptions,
    rewrite_english_to_cantonese_accent,
    rewrite_english_to_cantonese_accent_detailed,
)
from tools.cantonese_accent_english.formatting import (
    detokenize,
    format_cmu_phones,
    tokenize_text,
    validate_phones,
)
from tools.cantonese_accent_english.rules import apply_light_rules
from tools.cantonese_accent_english.rules_common import WordContext
import random


def _opts(**kwargs) -> AccentRewriteOptions:
    defaults = {"preset": "light", "seed": 1234}
    defaults.update(kwargs)
    return AccentRewriteOptions(**defaults)


class TestFormatting:
    def test_format_cmu_phones(self):
        assert format_cmu_phones(["F", "IH1", "NG", "K"]) == "[F][IH1][NG][K]"

    def test_validate_phones_rejects_empty(self):
        assert validate_phones([]) is False

    def test_validate_phones_requires_vowel(self):
        assert validate_phones(["T", "K"]) is False

    def test_detokenize_punctuation_spacing(self):
        out = detokenize(["Hello,", "[D][IH1][S]", "is", "good", "."])
        assert out == "Hello, [D][IH1][S] is good."

    def test_tokenize_sentence_start_flag(self):
        tokens = tokenize_text("Hello. The end.")
        words = [t for t in tokens if t.is_word]
        assert words[0].is_sentence_end is True
        assert words[1].is_sentence_end is True  # first word after "."
        assert words[2].is_sentence_end is False


class TestRules:
    def test_th_to_f_deterministic(self):
        rng = random.Random(0)
        result = apply_light_rules(
            ["TH", "IH1", "NG", "K"],
            WordContext(word="think"),
            rng=rng,
            disabled_rules=set(),
            enabled_rules={"TH_TO_F"},
        )
        assert result.phones == ["F", "IH1", "NG", "K"]
        assert "TH_TO_F" in result.applied_rules

    def test_dh_to_d_deterministic(self):
        rng = random.Random(0)
        result = apply_light_rules(
            ["DH", "IH1", "S"],
            WordContext(word="this"),
            rng=rng,
            disabled_rules=set(),
            enabled_rules={"DH_TO_D"},
        )
        assert result.phones == ["D", "IH1", "S"]

    def test_final_cluster_deletion(self):
        rng = random.Random(1)  # first draw < 0.7 for cluster deletion
        result = apply_light_rules(
            ["F", "R", "EH1", "N", "D"],
            WordContext(word="friend"),
            rng=rng,
            disabled_rules=set(),
            enabled_rules={
                "TH_TO_F",
                "DH_TO_D",
                "FINAL_CLUSTER_T_D_DELETION",
            },
        )
        assert result.phones == ["F", "R", "EH1", "N"]
        assert "FINAL_CLUSTER_T_D_DELETION" in result.applied_rules

    def test_final_cluster_keeps_vowel_before_t(self):
        rng = random.Random(0)
        result = apply_light_rules(
            ["B", "AE1", "D"],
            WordContext(word="bad"),
            rng=rng,
            disabled_rules=set(),
            enabled_rules={"FINAL_CLUSTER_T_D_DELETION"},
        )
        assert result.phones == ["B", "AE1", "D"]


class TestRewrite:
    def test_think_and_this_rewritten(self):
        text = "I think this is good."
        out = rewrite_english_to_cantonese_accent(text, _opts())
        assert "[F][IH1][NG][K]" in out
        assert "[D][IH1][S]" in out
        assert "think" not in out.lower() or "[F]" in out

    def test_strong_preset_works(self):
        out = rewrite_english_to_cantonese_accent(
            "think",
            AccentRewriteOptions(preset="strong", seed=1234),
        )
        assert "[" in out and "]" in out

    def test_deterministic_with_seed(self):
        text = "I think this video is very useful."
        a = rewrite_english_to_cantonese_accent(text, _opts(seed=1234))
        b = rewrite_english_to_cantonese_accent(text, _opts(seed=1234))
        assert a == b

    def test_unchanged_word_stays_plain(self):
        text = "I think."
        out = rewrite_english_to_cantonese_accent(text, _opts(seed=0))
        assert out.startswith("I ")
        assert "[AY1]" not in out

    def test_custom_override(self):
        text = "think"
        out = rewrite_english_to_cantonese_accent(
            text,
            _opts(custom_word_overrides={"think": ["S", "IH1", "NG", "K"]}),
        )
        assert out == "[S][IH1][NG][K]"

    def test_preserve_proper_noun_mid_sentence(self):
        text = "I met Alice yesterday."
        out = rewrite_english_to_cantonese_accent(text, _opts(seed=0))
        assert "Alice" in out

    def test_preserve_acronym(self):
        text = "The API is useful."
        out = rewrite_english_to_cantonese_accent(text, _opts(seed=0))
        assert "API" in out

    def test_max_rewrites_per_sentence_resets(self):
        text = "think this think. think this."
        detailed = rewrite_english_to_cantonese_accent_detailed(
            text,
            _opts(seed=0, max_rewrites_per_sentence=1),
        )
        assert len(detailed.changed_words) >= 2

    def test_milestone_sentence_seed_1234(self):
        text = (
            "I think this video is very useful, and my friend watched "
            "the last text."
        )
        out = rewrite_english_to_cantonese_accent(text, _opts(seed=1234))
        assert "[F][IH1][NG][K]" in out
        assert "[D][IH1][S]" in out
        assert "[F][R][EH1][N]" in out or "friend" in out

    def test_oov_word_unchanged(self):
        text = "I think xyzzyword."
        out = rewrite_english_to_cantonese_accent(text, _opts())
        assert "xyzzyword" in out

    def test_disabled_rule(self):
        text = "think"
        out = rewrite_english_to_cantonese_accent(
            text,
            _opts(disabled_rules={"TH_TO_F"}),
        )
        assert out == "think"
