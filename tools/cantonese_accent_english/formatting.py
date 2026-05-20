from __future__ import annotations

import re
from dataclasses import dataclass

# CMU/ARPAbet tokens compatible with CosyVoice3 bracket syntax.
VALID_CMU_TOKENS: frozenset[str] = frozenset({
    "AA", "AA0", "AA1", "AA2",
    "AE", "AE0", "AE1", "AE2",
    "AH", "AH0", "AH1", "AH2",
    "AO", "AO0", "AO1", "AO2",
    "AW", "AW0", "AW1", "AW2",
    "AY", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH",
    "EH", "EH0", "EH1", "EH2",
    "ER", "ER0", "ER1", "ER2",
    "EY", "EY0", "EY1", "EY2",
    "F", "G", "HH",
    "IH", "IH0", "IH1", "IH2",
    "IY", "IY0", "IY1", "IY2",
    "JH", "K", "L", "M", "N", "NG",
    "OW", "OW0", "OW1", "OW2",
    "OY", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH",
    "UH", "UH0", "UH1", "UH2",
    "UW", "UW0", "UW1", "UW2",
    "V", "W", "Y", "Z", "ZH",
})

_VOWEL_BASES = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
})

_NO_SPACE_BEFORE = frozenset(".,!?;:)]}")
_NO_SPACE_AFTER = frozenset("([{")

_TOKEN_RE = re.compile(
    r"\[([A-Z0-9]+)\]"  # existing bracket phoneme chains
    r"|[A-Za-z]+(?:'[A-Za-z]+)?"  # words with optional apostrophe
    r"|[^\s\w]"  # single punctuation / symbol
    r"|\s+",
)


@dataclass
class TextToken:
    text: str
    is_word: bool
    is_sentence_end: bool = False


def _phone_base(phone: str) -> str:
    return re.sub(r"\d+$", "", phone)


def has_vowel(phones: list[str]) -> bool:
    return any(_phone_base(p) in _VOWEL_BASES for p in phones)


def format_cmu_phones(phones: list[str]) -> str:
    return "".join(f"[{p}]" for p in phones)


def validate_phones(phones: list[str]) -> bool:
    if not phones:
        return False
    if not all(p in VALID_CMU_TOKENS for p in phones):
        return False
    if not has_vowel(phones):
        return False
    return True


def tokenize_text(text: str) -> list[TextToken]:
    tokens: list[TextToken] = []
    at_sentence_start = True
    for match in _TOKEN_RE.finditer(text):
        piece = match.group(0)
        if piece.isspace():
            continue
        if piece.startswith("["):
            tokens.append(TextToken(text=piece, is_word=False))
            continue
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", piece):
            tokens.append(
                TextToken(text=piece, is_word=True, is_sentence_end=at_sentence_start),
            )
            at_sentence_start = False
            continue
        tokens.append(TextToken(text=piece, is_word=False))
        if piece in ".?!":
            at_sentence_start = True
    return tokens


def detokenize(output_tokens: list[str]) -> str:
    parts: list[str] = []
    for piece in output_tokens:
        if not piece:
            continue
        if not parts:
            parts.append(piece)
            continue
        prev = parts[-1]
        if piece in _NO_SPACE_BEFORE or prev in _NO_SPACE_AFTER:
            parts.append(piece)
        else:
            parts.append(" " + piece)
    return "".join(parts)
