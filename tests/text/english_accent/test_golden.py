from tools.cantonese_accent_english import (
    AccentRewriteOptions,
    rewrite_english_to_cantonese_accent,
)


def test_golden_example1_light_with_overrides():
    text = "I think this video is very useful."
    out = rewrite_english_to_cantonese_accent(
        text,
        AccentRewriteOptions(
            preset="light",
            seed=1234,
            use_default_overrides=True,
        ),
    )
    assert "[F][IH1][NG][K]" in out
    assert "[D][IH1][S]" in out
    assert "[W][IH1][D][IY0][OW0]" in out
    assert "[W][EH1][R][IY0]" in out


def test_golden_example1_strong_with_overrides():
    text = "I think this video is very useful."
    out = rewrite_english_to_cantonese_accent(
        text,
        AccentRewriteOptions(
            preset="strong",
            seed=1234,
            use_default_overrides=True,
        ),
    )
    assert "[F][IH1][NG]" in out
    assert "[W][EH1][R][IY0]" in out
