"""Default high-impact word overrides from spec section 15."""

from __future__ import annotations

LIGHT_DEFAULT_OVERRIDES: dict[str, list[str]] = {
    "think": ["F", "IH1", "NG", "K"],
    "three": ["F", "R", "IY1"],
    "this": ["D", "IH1", "S"],
    "that": ["D", "AE1", "T"],
    "very": ["W", "EH1", "R", "IY0"],
    "video": ["W", "IH1", "D", "IY0", "OW0"],
}

STRONG_DEFAULT_OVERRIDES: dict[str, list[str]] = {
    "think": ["F", "IH1", "NG"],
    "three": ["F", "R", "IY1"],
    "this": ["D", "IH1", "S"],
    "that": ["D", "EH1"],
    "very": ["W", "EH1", "R", "IY0"],
    "video": ["W", "IH1", "D", "IY0", "OW0"],
    "girl": ["G", "ER1", "OW0"],
    "world": ["W", "ER1", "OW0"],
    "text": ["T", "EH1", "K", "S"],
    "last": ["L", "AE1", "S"],
}


def get_default_overrides(preset: str) -> dict[str, list[str]]:
    if preset == "light":
        return {k: list(v) for k, v in LIGHT_DEFAULT_OVERRIDES.items()}
    if preset == "strong":
        return {k: list(v) for k, v in STRONG_DEFAULT_OVERRIDES.items()}
    raise ValueError(f"Unknown preset: {preset!r}")
