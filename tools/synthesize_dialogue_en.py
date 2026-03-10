#!/usr/bin/env python3
"""
English Dialogue Speech Synthesis Tool

Reads dialogue text files (format: [Role] Spoken content) and synthesizes
English speech using CosyVoice, with different voice timbres per role.

Supported modes:
1. cross_lingual (default): Use any language speaker audio to synthesize English.
   Best when reference audio is Chinese or another non-English language.
2. zero_shot: Use English speaker audio with matching English prompt text.
   Best when reference audio is already in English.

Dialogue file format (same as Chinese version):
    [Doctor] Good morning, what brings you in today?
    [Patient] I've been having headaches for the past week.

Example usage:
    # Cross-lingual (Chinese speaker -> English speech)
    python tools/synthesize_dialogue_en.py \\
        --model-dir pretrained_models/Fun-CosyVoice3-0.5B \\
        --dialogue-dir data/dialogues_en \\
        --speaker-dir data/speakers \\
        --output-dir outputs/synthesized_en \\
        --mode cross_lingual \\
        --fp16

    # Zero-shot (English speaker -> English speech)
    python tools/synthesize_dialogue_en.py \\
        --model-dir pretrained_models/Fun-CosyVoice3-0.5B \\
        --dialogue-dir data/dialogues_en \\
        --speaker-dir data/speakers_en \\
        --output-dir outputs/synthesized_en \\
        --mode zero_shot \\
        --default-content "I hope you can do even better than me in the future." \\
        --fp16

    # Multi-process parallel
    python tools/synthesize_dialogue_en.py ... --worker-id 0 --num-workers 4 &
    python tools/synthesize_dialogue_en.py ... --worker-id 1 --num-workers 4 &
    python tools/synthesize_dialogue_en.py ... --worker-id 2 --num-workers 4 &
    python tools/synthesize_dialogue_en.py ... --worker-id 3 --num-workers 4 &
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio


ROLE_ALIASES = {
    # Medical
    "doctor": "doctor",
    "dr": "doctor",
    "dr.": "doctor",
    "physician": "doctor",
    "patient": "patient",
    "nurse": "nurse",
    "surgeon": "surgeon",
    "chief surgeon": "chief_surgeon",
    "anesthesiologist": "anesthesiologist",
    "resident": "resident",
    "intern": "intern",
    "attending": "attending",
    "attending doctor": "attending",
    # Interview
    "interviewer": "interviewer",
    "interviewee": "interviewee",
    "candidate": "interviewee",
    # Customer service
    "customer": "customer",
    "agent": "agent",
    "representative": "agent",
    "rep": "agent",
    # Education
    "teacher": "teacher",
    "professor": "teacher",
    "student": "student",
    # Media
    "host": "host",
    "guest": "guest",
    # Generic
    "speaker a": "speaker_a",
    "speaker b": "speaker_b",
    "speaker 1": "speaker_1",
    "speaker 2": "speaker_2",
    "speaker 3": "speaker_3",
    "speaker 4": "speaker_4",
    "a": "speaker_a",
    "b": "speaker_b",
}

ROLE_DISPLAY_NAMES = {
    "doctor": "Doctor",
    "patient": "Patient",
    "nurse": "Nurse",
    "surgeon": "Surgeon",
    "chief_surgeon": "Chief Surgeon",
    "anesthesiologist": "Anesthesiologist",
    "resident": "Resident",
    "intern": "Intern",
    "attending": "Attending Doctor",
    "interviewer": "Interviewer",
    "interviewee": "Interviewee",
    "customer": "Customer",
    "agent": "Agent",
    "teacher": "Teacher",
    "student": "Student",
    "host": "Host",
    "guest": "Guest",
    "speaker_a": "Speaker A",
    "speaker_b": "Speaker B",
    "speaker_1": "Speaker 1",
    "speaker_2": "Speaker 2",
    "speaker_3": "Speaker 3",
    "speaker_4": "Speaker 4",
}


def normalize_role(role_text: str) -> str:
    """
    Normalize role text to a standard identifier.

    Supports exact match, alias lookup, and fallback to sanitized original text
    for arbitrary custom roles.
    """
    cleaned = role_text.strip()
    lower = cleaned.lower()

    if lower in ROLE_ALIASES:
        return ROLE_ALIASES[lower]

    sanitized = re.sub(r'[^a-z0-9]+', '_', lower).strip('_')
    return sanitized if sanitized else "unknown"


def parse_dialogue_file(filepath: Path) -> List[Dict[str, str]]:
    """
    Parse a dialogue text file.

    Format: [Role] Spoken content (one utterance per line).
    Returns: [{"role": "doctor", "role_display": "Doctor", "text": "..."}, ...]
    """
    lines = filepath.read_text(encoding="utf-8").strip().splitlines()
    utterances = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"\[([^\]]+)\]\s*(.+)", line)
        if not match:
            continue

        role_text, content = match.groups()
        role_text = role_text.strip()
        content = content.strip()

        role = normalize_role(role_text)
        role_display = ROLE_DISPLAY_NAMES.get(role, role_text)

        utterances.append({
            "role": role,
            "role_display": role_display,
            "text": content,
        })

    print(f"[DEBUG] Parsed {len(utterances)} utterances from {filepath.name}")
    for i, utt in enumerate(utterances[:5]):
        preview = utt['text'][:50] + '...' if len(utt['text']) > 50 else utt['text']
        print(f"        [{i}] {utt['role']} ({utt['role_display']}): {preview}")

    return utterances


def extract_unique_roles(utterances: List[Dict[str, str]]) -> List[str]:
    """Extract unique roles preserving first-appearance order."""
    seen = set()
    unique = []
    for utt in utterances:
        if utt["role"] not in seen:
            seen.add(utt["role"])
            unique.append(utt["role"])
    return unique


def load_speaker_prompt_text(
    speaker_wav: Path, prompt_prefix: str, default_content: str = ""
) -> str:
    """
    Load prompt text from a companion .txt file for the given speaker wav.

    E.g. speaker_001.wav -> speaker_001.txt
    Returns prompt_prefix + file_content (or default_content if no .txt found).
    """
    txt_file = speaker_wav.with_suffix('.txt')
    content = default_content

    if txt_file.exists():
        try:
            file_content = txt_file.read_text(encoding='utf-8').strip()
            if file_content:
                content = file_content
        except Exception as e:
            print(f"[WARNING] Failed to read text file: {txt_file.name} - {e}")

    return prompt_prefix + content


def select_speakers(
    speaker_files: List[Path], num_speakers: int, seed: Optional[int] = None
) -> List[Path]:
    """Randomly select N distinct speaker audio files."""
    if len(speaker_files) < num_speakers:
        raise ValueError(
            f"Speaker count ({len(speaker_files)}) < required ({num_speakers})"
        )
    rng = random.Random(seed) if seed is not None else random
    return rng.sample(speaker_files, num_speakers)


def normalize_english_text(text: str) -> str:
    """
    Basic English text normalization for TTS.

    Spell out common abbreviations and normalize punctuation.
    Heavy normalization (numbers, etc.) is handled by CosyVoice frontend
    when text_frontend=True.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize common quote styles
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    return text


def synthesize_dialogue(
    cosyvoice,
    utterances: List[Dict[str, str]],
    speaker_mapping: Dict[str, Path],
    output_dir: Path,
    dialogue_id: str,
    mode: str = "cross_lingual",
    text_prefix: str = "You are a helpful assistant.<|endofprompt|>",
    speaker_prompt_mapping: Optional[Dict[str, str]] = None,
    speed: float = 1.0,
    sample_rate: int = 22050,
) -> List[Path]:
    """
    Synthesize all utterances in a dialogue.

    For cross_lingual mode, text_prefix is prepended to each utterance text
    before tokenization (e.g. "<|en|>" or "You are a helpful assistant.<|endofprompt|>").

    For zero_shot mode, speaker_prompt_mapping must provide English prompt text
    per role.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    cached_prompts = {}

    for idx, utt in enumerate(utterances):
        role = utt["role"]
        text = utt["text"]
        speaker_wav = speaker_mapping.get(role)

        if not speaker_wav or not speaker_wav.exists():
            print(f"[WARNING] Role '{role}' has no speaker file, skipping: {text[:30]}...")
            continue

        output_file = output_dir / f"{dialogue_id}_{idx:03d}_{role}.wav"

        if output_file.exists() and output_file.stat().st_size > 1000:
            print(f"[SKIP] Already exists: {output_file.name}")
            generated_files.append(output_file)
            continue

        if role not in cached_prompts:
            try:
                print(f"[INFO] Loading speaker voice for '{role}': {speaker_wav.name}")

                if mode == "cross_lingual":
                    cached_prompts[role] = cosyvoice.frontend.frontend_cross_lingual(
                        "", str(speaker_wav), sample_rate, ""
                    )
                elif mode == "zero_shot":
                    prompt_text = (
                        speaker_prompt_mapping.get(role, "")
                        if speaker_prompt_mapping else ""
                    )
                    if not prompt_text:
                        print(f"[ERROR] zero_shot mode requires prompt text for role '{role}'")
                        continue
                    print(f"       prompt text: {prompt_text[:60]}...")
                    cached_prompts[role] = cosyvoice.frontend.frontend_zero_shot(
                        "", prompt_text, str(speaker_wav), sample_rate, ""
                    )
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                cached_prompts[role].pop("text", None)
                cached_prompts[role].pop("text_len", None)
                print(f"[INFO] Speaker voice for '{role}' loaded successfully")

            except Exception as e:
                print(f"[ERROR] Failed to load speaker voice for '{role}': {e}")
                import traceback
                traceback.print_exc()
                continue

        if role not in cached_prompts:
            continue

        try:
            model_input = dict(cached_prompts[role])

            synth_text = text_prefix + normalize_english_text(text)
            txt_token, txt_len = cosyvoice.frontend._extract_text_token(synth_text)
            model_input["text"] = txt_token
            model_input["text_len"] = txt_len

            for out in cosyvoice.model.tts(**model_input, stream=False, speed=speed):
                wav = out["tts_speech"]
                torchaudio.save(str(output_file), wav, sample_rate)
                print(f"[OK] {output_file.name} ({wav.shape[1] / sample_rate:.2f}s)")
                generated_files.append(output_file)
                break

        except Exception as e:
            print(f"[ERROR] Synthesis failed: {output_file.name} - {e}")
            import traceback
            traceback.print_exc()

    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="English dialogue speech synthesis tool (multi-process parallel)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-dir", type=str, required=True,
                        help="CosyVoice model directory")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable fp16 acceleration")
    parser.add_argument("--load-vllm", action="store_true",
                        help="Enable vLLM (if supported)")
    parser.add_argument("--mode", type=str, default="cross_lingual",
                        choices=["cross_lingual", "zero_shot"],
                        help="Synthesis mode: cross_lingual uses any-language speaker "
                             "audio; zero_shot requires English speaker audio with "
                             "matching English prompt text")
    parser.add_argument("--text-prefix", type=str, default=None,
                        help="Text prefix for synthesis. Defaults to "
                             "'You are a helpful assistant.<|endofprompt|>' for "
                             "CosyVoice3, '<|en|>' for CosyVoice1")
    parser.add_argument("--prompt-prefix", type=str,
                        default="You are a helpful assistant.<|endofprompt|>",
                        help="Prompt prefix for zero_shot mode (prepended to speaker "
                             "prompt text)")
    parser.add_argument("--default-content", type=str,
                        default="I hope you can do even better than me in the future.",
                        help="Default prompt text content when speaker .txt file is missing "
                             "(only used in zero_shot mode)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Intra-process concurrency (recommend 1~2)")

    parser.add_argument("--dialogue-dir", type=str, required=True,
                        help="Directory containing dialogue text files")
    parser.add_argument("--speaker-dir", type=str, required=True,
                        help="Directory containing speaker reference audio (.wav)")
    parser.add_argument("--output-dir", type=str, default="outputs/synthesized_en",
                        help="Output directory")

    parser.add_argument("--worker-id", type=int, default=0,
                        help="Worker ID (0-based) for multi-process sharding")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of workers")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible speaker assignment")

    parser.add_argument("--dialogue-pattern", type=str, default="*.txt",
                        help="Glob pattern for dialogue files")
    parser.add_argument("--speaker-pattern", type=str, default="*.wav",
                        help="Glob pattern for speaker files")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.append(str(repo_root / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel

    dialogue_dir = Path(args.dialogue_dir)
    speaker_dir = Path(args.speaker_dir)
    output_dir = Path(args.output_dir)

    if not dialogue_dir.exists():
        raise FileNotFoundError(f"Dialogue directory not found: {dialogue_dir}")
    if not speaker_dir.exists():
        raise FileNotFoundError(f"Speaker directory not found: {speaker_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[INFO] Loading model: {args.model_dir}")
    model_kwargs = {
        "model_dir": args.model_dir,
        "fp16": args.fp16,
        "load_vllm": args.load_vllm,
        "load_trt": False,
        "trt_concurrent": 1,
        "load_jit": False,
    }
    while True:
        try:
            cosyvoice = AutoModel(**model_kwargs)
            break
        except TypeError as e:
            key = None
            if "unexpected keyword argument" in str(e) and "'" in str(e):
                key = str(e).split("'")[1]
            if key and key in model_kwargs:
                model_kwargs.pop(key)
                continue
            raise

    os.environ["COSYVOICE_DISABLE_CUDA_CLEANUP"] = "1"
    try:
        cosyvoice.model.disable_cuda_cleanup = True
    except Exception:
        pass

    # Auto-detect model type for default text_prefix
    model_class = cosyvoice.__class__.__name__
    print(f"[INFO] Model loaded: {model_class}")

    if args.text_prefix is not None:
        text_prefix = args.text_prefix
    elif "CosyVoice3" in model_class or "CosyVoice2" in model_class:
        text_prefix = "You are a helpful assistant.<|endofprompt|>"
    else:
        text_prefix = "<|en|>"
    print(f"[INFO] Text prefix: {text_prefix!r}")
    print(f"[INFO] Synthesis mode: {args.mode}")

    dialogue_files = sorted(dialogue_dir.glob(args.dialogue_pattern))
    if not dialogue_files:
        raise ValueError(
            f"No matching dialogue files found: {dialogue_dir / args.dialogue_pattern}"
        )

    speaker_files = sorted(speaker_dir.glob(args.speaker_pattern))
    if not speaker_files:
        raise ValueError(
            f"No matching speaker files found: {speaker_dir / args.speaker_pattern}"
        )

    if len(speaker_files) < 2:
        raise ValueError(
            f"Speaker count ({len(speaker_files)}) < 2, cannot assign voices"
        )

    print(f"[INFO] Dialogue files: {len(dialogue_files)}")
    print(f"[INFO] Speaker files: {len(speaker_files)}")

    # Validate speaker files
    print("[INFO] Validating speaker files...")
    invalid_speakers = []
    for spk_file in speaker_files[:min(5, len(speaker_files))]:
        try:
            wav, sr = torchaudio.load(str(spk_file))
            duration = wav.shape[1] / sr
            if duration > 30:
                print(f"[WARNING] Speaker file exceeds 30s: {spk_file.name} ({duration:.2f}s)")
                invalid_speakers.append(spk_file)
            elif duration < 2:
                print(f"[WARNING] Speaker file too short: {spk_file.name} ({duration:.2f}s)")
        except Exception as e:
            print(f"[WARNING] Cannot read speaker file: {spk_file.name} - {e}")
            invalid_speakers.append(spk_file)

    if invalid_speakers:
        print(f"[WARNING] Found {len(invalid_speakers)} problematic speaker files")

    # Multi-process sharding
    worker_id = args.worker_id
    num_workers = args.num_workers
    my_dialogues = [f for i, f in enumerate(dialogue_files) if i % num_workers == worker_id]

    print(f"[INFO] Worker {worker_id}/{num_workers}, handling {len(my_dialogues)} dialogues")

    total_dialogues = len(my_dialogues)
    total_utterances = 0
    total_success = 0

    for idx, dialogue_file in enumerate(my_dialogues, 1):
        dialogue_id = dialogue_file.stem
        print(f"\n[{idx}/{total_dialogues}] Processing: {dialogue_id}")

        try:
            utterances = parse_dialogue_file(dialogue_file)
            if not utterances:
                print(f"[WARNING] Empty or malformed dialogue, skipping: {dialogue_file.name}")
                continue

            print(f"  - Utterances: {len(utterances)}")
            total_utterances += len(utterances)

            unique_roles = extract_unique_roles(utterances)
            num_roles = len(unique_roles)

            if num_roles == 0:
                print(f"[WARNING] No valid roles found, skipping: {dialogue_file.name}")
                continue

            if len(speaker_files) < num_roles:
                print(
                    f"[WARNING] Not enough speakers ({len(speaker_files)}) "
                    f"for {num_roles} roles, skipping: {dialogue_file.name}"
                )
                continue

            print(f"  - Roles ({num_roles}): {unique_roles}")

            dialogue_seed = args.seed + hash(dialogue_id) % 100000
            selected_speakers = select_speakers(
                speaker_files, num_speakers=num_roles, seed=dialogue_seed
            )

            speaker_mapping = dict(zip(unique_roles, selected_speakers))

            print("  - Speaker assignment:")
            for role, spk_path in speaker_mapping.items():
                display = ROLE_DISPLAY_NAMES.get(role, role)
                print(f"    {display} ({role}): {spk_path.name}")

            # Build prompt text mapping (only needed for zero_shot mode)
            speaker_prompt_mapping = None
            if args.mode == "zero_shot":
                speaker_prompt_mapping = {}
                for role, spk_wav in speaker_mapping.items():
                    prompt_text = load_speaker_prompt_text(
                        spk_wav, args.prompt_prefix, args.default_content
                    )
                    speaker_prompt_mapping[role] = prompt_text

            generated = synthesize_dialogue(
                cosyvoice=cosyvoice,
                utterances=utterances,
                speaker_mapping=speaker_mapping,
                output_dir=output_dir / dialogue_id,
                dialogue_id=dialogue_id,
                mode=args.mode,
                text_prefix=text_prefix,
                speaker_prompt_mapping=speaker_prompt_mapping,
                speed=args.speed,
                sample_rate=cosyvoice.sample_rate,
            )

            total_success += len(generated)
            print(f"  - Synthesized: {len(generated)} audio files")

            meta_file = output_dir / dialogue_id / f"{dialogue_id}_meta.json"
            meta = {
                "dialogue_id": dialogue_id,
                "source_file": str(dialogue_file),
                "language": "en",
                "mode": args.mode,
                "text_prefix": text_prefix,
                "speaker_mapping": {k: str(v) for k, v in speaker_mapping.items()},
                "utterances": utterances,
                "generated_files": [str(f) for f in generated],
            }
            meta_file.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        except Exception as e:
            print(f"[ERROR] Failed to process dialogue: {dialogue_file.name} - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"[DONE] Worker {worker_id}/{num_workers}")
    print(f"  - Dialogues processed: {total_dialogues}")
    print(f"  - Total utterances: {total_utterances}")
    print(f"  - Successfully synthesized: {total_success} audio files")
    print(f"  - Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
