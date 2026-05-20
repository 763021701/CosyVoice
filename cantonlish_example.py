#!/usr/bin/env python3
"""
Compare plain vs light (weak) vs strong Cantonese-accented English TTS.

Usage:
    conda run -n cosyvoice python example.py \
        --text "I think this video is very useful."

    conda run -n cosyvoice python example.py \
        --text-file my_script.txt \
        --model-dir FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
        --prompt-wav ./asset/zero_shot_prompt.wav \
        --output-prefix outputs/accent_compare
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, "third_party", "Matcha-TTS"))

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tools.cantonese_accent_english import (
    AccentRewriteOptions,
    rewrite_english_to_cantonese_accent,
    rewrite_english_to_cantonese_accent_detailed,
)


DEFAULT_MODEL_DIR = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_PROMPT_WAV = "/root/autodl-tmp/workspace/dataset/Test_Samples/Sample1_0139_0149.wav"
DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "Specimen labelled uterus and bilateral ovaries and tubes.  "
    "Submitted was a hysterectomy and bilateral salpingo-oophorectomy specimen."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize plain / light accent / strong accent English for comparison.",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text",
        type=str,
        help="English text to synthesize (plain input; accent rewrite applied automatically).",
    )
    text_group.add_argument(
        "--text-file",
        type=str,
        help="Path to a UTF-8 text file (contents used as TTS input).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"CosyVoice3 model directory or ModelScope id (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--prompt-wav",
        type=str,
        default=DEFAULT_PROMPT_WAV,
        help="Reference speaker wav for zero-shot cloning.",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=DEFAULT_PROMPT_TEXT,
        help="Prompt text paired with --prompt-wav.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for accent rewrite (light/strong).",
    )
    parser.add_argument(
        "--use-default-overrides",
        action="store_true",
        help="Apply built-in high-impact word overrides from the accent module.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="accent_compare",
        help="Output wav prefix: {prefix}_{plain,weak,strong}_0.wav",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load model in fp16 (default: fp32).",
    )
    return parser.parse_args()


def _load_input_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text.strip()
    with open(args.text_file, encoding="utf-8") as f:
        return f.read().strip()


def _build_variants(
    plain_text: str,
    *,
    seed: int,
    use_default_overrides: bool,
) -> list[tuple[str, str]]:
    weak = rewrite_english_to_cantonese_accent(
        plain_text,
        AccentRewriteOptions(
            preset="light",
            seed=seed,
            use_default_overrides=use_default_overrides,
        ),
    )
    strong = rewrite_english_to_cantonese_accent(
        plain_text,
        AccentRewriteOptions(
            preset="strong",
            seed=seed,
            use_default_overrides=use_default_overrides,
        ),
    )
    return [
        ("plain", plain_text),
        ("weak", weak),
        ("strong", strong),
    ]


def _print_rewrite_summary(plain_text: str, seed: int, use_default_overrides: bool) -> None:
    for preset in ("light", "strong"):
        detailed = rewrite_english_to_cantonese_accent_detailed(
            plain_text,
            AccentRewriteOptions(
                preset=preset,
                seed=seed,
                use_default_overrides=use_default_overrides,
            ),
        )
        label = "weak (light)" if preset == "light" else "strong"
        print(f"--- {label} | changed {len(detailed.changed_words)} words ---")
        print(detailed.rewritten_text)
        print()


def main() -> None:
    args = _parse_args()
    plain_text = _load_input_text(args)

    print("=== input (plain) ===")
    print(plain_text)
    print()

    variants = _build_variants(
        plain_text,
        seed=args.seed,
        use_default_overrides=args.use_default_overrides,
    )

    _print_rewrite_summary(plain_text, args.seed, args.use_default_overrides)

    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_trt=True,
        load_vllm=True,
        fp16=args.fp16,
    )

    for variant_name, tts_text in variants:
        set_all_random_seed(args.seed)
        print(f"=== synthesizing | {variant_name} ===")
        print(tts_text)
        print()
        for i, chunk in enumerate(
            cosyvoice.inference_zero_shot(
                tts_text,
                args.prompt_text,
                args.prompt_wav,
                stream=False,
                text_frontend=False,
            )
        ):
            out_path = f"{args.output_prefix}_{variant_name}_{i}.wav"
            torchaudio.save(out_path, chunk["tts_speech"], cosyvoice.sample_rate)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()
