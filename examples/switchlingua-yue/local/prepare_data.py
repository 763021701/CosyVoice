#!/usr/bin/env python3
"""SwitchLingua Cantonese -> Kaldi-style dirs, split 8:1:1 disjoint by text-item.

Cantonese.csv file_name is '{item}_{sentence}': one LLM-generated story yields several sentences, so
a per-utterance random split would put neighbours of the same story in train and test at once. We
split by item. 26 texts are recorded more than once (92 clips); items sharing an exact normalized
text are merged into one group first so a line cannot straddle splits either.

The dataset ships no speaker id (checked: no speaker column in Cantonese/Mandarin/Arabic/Japanese),
so spk = utt and spk_embedding == utt_embedding -- which is also
cosyvoice/dataset/processor.parse_embedding's own fallback when no embedding column is present.

The CSV/wav are downloaded data: run with -I and pass their paths as arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
SPLITS = ("train", "dev", "test")


def load_rows(csv_path: Path, wav_dir: Path) -> list[dict]:
    csv.field_size_limit(10 ** 9)
    rows = []
    for r in csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")):
        stem = r["file_name"].rsplit(".", 1)[0]
        wav = wav_dir / f"{stem}.wav"
        text = " ".join((r["text"] or "").split())
        if not text or not wav.is_file():
            continue
        rows.append({"id": stem, "item": stem.split("_")[0], "text": text,
                     "wav": str(wav), "cs_ratio": r["cs_ratio"]})
    return rows


def build_groups(rows: list[dict]) -> list[list[str]]:
    """Item ids, merged into one group when they share an exact text, sorted for stable output."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen_text: dict[str, str] = {}
    for r in rows:
        find(r["item"])
        if r["text"] in seen_text:
            parent[find(r["item"])] = find(seen_text[r["text"]])
        else:
            seen_text[r["text"]] = find(r["item"])
    buckets: dict[str, list[str]] = defaultdict(list)
    for item in list(parent):
        buckets[find(item)].append(item)
    return sorted((sorted(v) for v in buckets.values()))


def write_split(out_root: Path, name: str, rows: list[dict]) -> dict:
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["id"])
    (d / "wav.scp").write_text("".join(f"{r['id']} {r['wav']}\n" for r in rows))
    (d / "text").write_text("".join(f"{r['id']} {r['text']}\n" for r in rows))
    (d / "utt2spk").write_text("".join(f"{r['id']} {r['id']}\n" for r in rows))
    (d / "instruct").write_text("".join(f"{r['id']} {INSTRUCT}\n" for r in rows))
    (d / "spk2utt").write_text("".join(f"{r['id']} {r['id']}\n" for r in rows))
    ratio = defaultdict(int)
    for r in rows:
        ratio[r["cs_ratio"]] += 1
    return {"n_utt": len(rows), "n_item": len({r["item"] for r in rows}),
            "cs_ratio": dict(sorted(ratio.items()))}


def read_ids(path: Path) -> set[str]:
    return {l.split()[0] for l in path.read_text().splitlines()}


def read_texts(path: Path) -> set[str]:
    return {l.split(maxsplit=1)[1] for l in path.read_text().splitlines()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ratios", default="8,1,1")
    ap.add_argument("--seed", type=int, default=1986)
    args = ap.parse_args()

    rows = load_rows(args.csv, args.wav_dir)
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item"]].append(r)
    groups = build_groups(rows)
    rng = random.Random(args.seed)
    groups.sort(key=lambda g: (rng.random(), g[0]))

    ratios = [int(x) for x in args.ratios.split(",")]
    bounds = [round(len(groups) * sum(ratios[:k]) / sum(ratios)) for k in range(len(ratios) + 1)]
    report = {"csv": str(args.csv), "wav_dir": str(args.wav_dir), "seed": args.seed,
              "ratios": ratios, "instruct": INSTRUCT, "splits": {}}
    print(f"utt {len(rows)}  item {len(by_item)}  split-group {len(groups)}")
    for k, name in enumerate(SPLITS):
        items = [i for g in groups[bounds[k]:bounds[k + 1]] for i in g]
        sub = [r for i in items for r in by_item[i]]
        report["splits"][name] = write_split(args.out, name, sub)
        print(f"  {name:5s} {report['splits'][name]}")

    print("leakage check (must all be 0):")
    ok = True
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            a, b = SPLITS[i], SPLITS[j]
            leak = {"item": len({x.split("_")[0] for x in read_ids(args.out / a / "wav.scp")} &
                                 {x.split("_")[0] for x in read_ids(args.out / b / "wav.scp")}),
                    "utt": len(read_ids(args.out / a / "wav.scp") & read_ids(args.out / b / "wav.scp")),
                    "text": len(read_texts(args.out / a / "text") & read_texts(args.out / b / "text"))}
            ok &= not any(leak.values())
            report["splits"][a][f"overlap_{b}"] = leak
            print(f"  {a}/{b}: {leak}")
    assert ok, "leakage across splits"

    report["total_utt"] = sum(s["n_utt"] for s in report["splits"].values())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {args.out / 'split_report.json'}")


if __name__ == "__main__":
    main()
