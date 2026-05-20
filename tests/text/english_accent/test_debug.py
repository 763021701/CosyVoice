from tools.cantonese_accent_english import (
    AccentRewriteOptions,
    format_debug_report,
    rewrite_english_to_cantonese_accent_detailed,
)


def test_format_debug_report():
    result = rewrite_english_to_cantonese_accent_detailed(
        "think",
        AccentRewriteOptions(preset="light", seed=1234),
    )
    report = format_debug_report(result)
    assert len(report) >= 1
    entry = report[0]
    assert entry["word"] == "think"
    assert entry["canonical"] == ["TH", "IH1", "NG", "K"]
    assert entry["rewritten"] == ["F", "IH1", "NG", "K"]
    assert "TH_TO_F" in entry["rules"]
