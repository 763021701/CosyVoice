#!/usr/bin/env python3
"""Assemble CosyVoice training parquet from a Kaldi dir + the .pt files written by
tools/extract_embedding.py and tools/extract_speech_token.py.

tools/make_parquet_list.py cannot be used: its job() writes spk2parquet from `spk_list`, a name that is
never defined anywhere in the file (line 54), so every shard raises NameError in the Pool worker. It also
forks a Pool after importing torch. Columns here match job()'s exactly -- and nothing else is needed:
grep shows no reader of spk2parquet/utt2parquet outside that tool.

utt2spk is the identity here (the dataset has no speaker id), so spk_embedding == utt_embedding, which is
also processor.parse_embedding's fallback. Padding consumes both unconditionally, so both must be present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def read_scp(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        k, v = line.split(maxsplit=1)
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="Kaldi dir holding the .pt files")
    ap.add_argument("--per_file", type=int, default=1000)
    args = ap.parse_args()

    d = args.dir
    utt2wav = read_scp(d / "wav.scp")
    utt2text = read_scp(d / "text")
    utt2spk = read_scp(d / "utt2spk")
    utt2instruct = read_scp(d / "instruct")
    utt2embedding = torch.load(d / "utt2embedding.pt", map_location="cpu", weights_only=True)
    spk2embedding = torch.load(d / "spk2embedding.pt", map_location="cpu", weights_only=True)
    utt2speech_token = torch.load(d / "utt2speech_token.pt", map_location="cpu", weights_only=True)

    utt_list = list(utt2wav)  # wav.scp order
    for name, table in (("utt2embedding", utt2embedding), ("utt2speech_token", utt2speech_token)):
        missing = [u for u in utt_list if u not in table]
        assert not missing, f"{name} missing {len(missing)} utt, e.g. {missing[:3]}"
    empty = [u for u in utt_list if len(utt2speech_token[u]) == 0]
    assert not empty, f"empty speech_token {len(empty)} utt, e.g. {empty[:3]}"

    out = d / "parquet"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(0, len(utt_list), args.per_file):
        chunk = utt_list[i:i + args.per_file]
        path = out / f"parquet_{i:09d}.tar"
        df = pd.DataFrame()
        df["utt"] = chunk
        df["audio_data"] = [Path(utt2wav[u]).read_bytes() for u in chunk]
        df["wav"] = [utt2wav[u] for u in chunk]
        df["text"] = [utt2text[u] for u in chunk]
        df["spk"] = [utt2spk[u] for u in chunk]
        df["utt_embedding"] = [utt2embedding[u] for u in chunk]
        df["spk_embedding"] = [spk2embedding[utt2spk[u]] for u in chunk]
        df["speech_token"] = [utt2speech_token[u] for u in chunk]
        df["instruct"] = [utt2instruct[u] for u in chunk]
        df.to_parquet(path)
        files.append(str(path.resolve()))
    (out / "data.list").write_text("".join(f"{p}\n" for p in files))
    print(f"{d.name}: {len(utt_list)} utt -> {len(files)} parquet, {out / 'data.list'}")


if __name__ == "__main__":
    main()
