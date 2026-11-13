#!/usr/bin/env python3
"""Build a continuation parquet: each row is one speaker's next utterance given another of theirs.

The stock parquet (local/make_parquet.py) trains p(speech tokens | text). Every speaker of one accent
is then a draw from the same distribution, and sampling only jitters inside that average. Zero-shot
inference does not: CosyVoice3LM.inference feeds [prompt text ++ target text, task id, prompt speech
tokens] and generates the target tokens. This file packs each training row the same way, so the loss
on the target tokens is conditioned on that speaker's own prompt tokens.

SwitchLingua writes utt2spk as the identity (prepare_data.py). Speakers are recovered from the item
prefix of the utt id, the same key build_prompt_pairs.py uses. A real utt2spk with groups of size
>= 2 is used instead when one exists.

The trainer still prepends the `instruct` column and then tokenizes `text` as one string, and it
drops a row whose text tokens exceed conf/cosyvoice3.yaml filter.token_max_length (200). Rows over
that cap, or whose speech tokens exceed --max-side-tokens on either side, are skipped and counted.

    python local/make_continuation_parquet.py --dir data/train --out data_continuation/train
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "third_party" / "Matcha-TTS")]

from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer  # noqa: E402

INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
# conf/cosyvoice3.yaml filter.token_max_length. The trainer drops anything longer with no log line.
TEXT_TOKEN_MAX = 200


def read_scp(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        k, v = line.split(maxsplit=1)
        out[k] = v
    return out


def as_list(value) -> list:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def speaker_groups(utts: list[str], utt2spk: dict) -> tuple[dict, str]:
    by_spk = defaultdict(list)
    for utt in utts:
        by_spk[utt2spk[utt]].append(utt)
    if max((len(v) for v in by_spk.values()), default=0) >= 2:
        return by_spk, "utt2spk"
    by_item = defaultdict(list)
    for utt in utts:
        by_item[utt.split("_", 1)[0]].append(utt)
    return by_item, "item_prefix"


def join_text(prompt: str, target: str) -> str:
    if prompt and target and not prompt[-1].isspace() and not target[0].isspace():
        return prompt + " " + target
    return prompt + target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="Kaldi dir with wav.scp, text, tokens, embeddings")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pretrained", type=Path,
                    default=Path("/root/autodl-tmp/.cache/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    ap.add_argument("--per-file", type=int, default=1000)
    ap.add_argument("--prompts-per-target", type=int, default=1,
                    help="how many other utterances of the same speaker to pair with each target")
    ap.add_argument("--max-side-tokens", type=int, default=750,
                    help="drop a side longer than this many speech tokens (750 = 30 s at 25 Hz, the "
                         "inference prompt cap in frontend.py)")
    ap.add_argument("--max-pairs", type=int, default=0, help="stop after this many kept pairs; 0 = all")
    args = ap.parse_args()

    d = args.dir
    utt2wav = read_scp(d / "wav.scp")
    utt2text = read_scp(d / "text")
    utt2spk = read_scp(d / "utt2spk")
    utt2embedding = torch.load(d / "utt2embedding.pt", map_location="cpu", weights_only=True)
    spk2embedding = torch.load(d / "spk2embedding.pt", map_location="cpu", weights_only=True)
    utt2speech_token = torch.load(d / "utt2speech_token.pt", map_location="cpu", weights_only=True)
    tokenizer = get_qwen_tokenizer(str(args.pretrained / "CosyVoice-BlankEN"), True, "cosyvoice3")

    utts = list(utt2wav)
    groups, how = speaker_groups(utts, utt2spk)
    usable = {k: sorted(v) for k, v in groups.items() if len(v) >= 2}
    skipped_singleton = sum(len(v) for v in groups.values() if len(v) < 2)
    print(f"speakers via {how}: {len(usable)} groups with >=2 utts, {skipped_singleton} singleton utts skipped")

    rows = []
    n_token = n_text = 0
    for members in usable.values():
        n = len(members)
        for i, target in enumerate(members):
            for k in range(args.prompts_per_target):
                prompt = members[(i + 1 + k) % n]
                if prompt == target:
                    continue
                prompt_tok = as_list(utt2speech_token[prompt])
                target_tok = as_list(utt2speech_token[target])
                if not prompt_tok or not target_tok:
                    n_token += 1
                    continue
                if len(prompt_tok) > args.max_side_tokens or len(target_tok) > args.max_side_tokens:
                    n_token += 1
                    continue
                text = join_text(utt2text[prompt], utt2text[target])
                # The trainer encodes instruct and text separately, then concatenates. Count both.
                n_ids = len(tokenizer.encode(INSTRUCT, allowed_special="all")) + len(tokenizer.encode(text, allowed_special="all"))
                if n_ids > TEXT_TOKEN_MAX:
                    n_text += 1
                    continue
                spk = utt2spk[target]
                rows.append({
                    "utt": f"{target}__from__{prompt}",
                    "wav": utt2wav[target],
                    "text": text,
                    "spk": spk,
                    "utt_embedding": as_list(utt2embedding[target]),
                    "spk_embedding": as_list(spk2embedding[spk]),
                    "speech_token": prompt_tok + target_tok,
                    "instruct": INSTRUCT,
                    "prompt_utt": prompt,
                    "n_prompt_token": len(prompt_tok),
                })
                if args.max_pairs and len(rows) >= args.max_pairs:
                    break
            if args.max_pairs and len(rows) >= args.max_pairs:
                break
        if args.max_pairs and len(rows) >= args.max_pairs:
            break

    assert rows, "no pairs kept"
    args.out.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(0, len(rows), args.per_file):
        chunk = rows[i:i + args.per_file]
        path = args.out / f"parquet_{i:09d}.tar"
        df = pd.DataFrame(chunk)
        df["audio_data"] = [Path(p).read_bytes() for p in df["wav"]]
        df.to_parquet(path)
        files.append(str(path.resolve()))
    (args.out / "data.list").write_text("".join(f"{p}\n" for p in files))
    print(f"kept {len(rows)} pairs -> {len(files)} parquet, skipped {n_token} for speech-token length, "
          f"{n_text} for text-token length > {TEXT_TOKEN_MAX}")
    print(f"data.list {args.out / 'data.list'}")


if __name__ == "__main__":
    main()
