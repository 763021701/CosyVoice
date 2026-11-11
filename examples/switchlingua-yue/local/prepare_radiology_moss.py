#!/usr/bin/env python3
"""Radiology MossFormer2 cuts -> one Kaldi train dir. Same text filters as prepare_mix_data.

The corpus is a few tenths of an hour, so every kept clip goes to train. dev/test are written empty.
Silence trimming is a separate pass: local/clean_pathology.py, the same one that built data_mix_clean.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_mix_data import INSTRUCT, unusable

NAMES = ("train", "dev", "test")
DEFAULT_CSV = Path(
    "/root/autodl-tmp/workspace/dataset/Test_Samples/samples/radiology/output_mossformer2/metadata.csv"
)


def rows_from_csv(path: Path) -> list[dict]:
    out = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            doctor = raw["source"].split("/")[0]
            stem = Path(raw["audio_path"]).stem
            out.append(
                {
                    "utt": f"{doctor}_{stem}",
                    "case": doctor,
                    "wav": raw["audio_path"],
                    "text": " ".join(raw["text"].split()),
                    "dur": float(raw["duration"]),
                    "punct": 0,
                    "src": "radiology_mossformer2",
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    kept, dropped = [], collections.Counter()
    for row in rows_from_csv(args.csv):
        why = unusable(row)
        dropped[why or "kept"] += 1
        if why is None:
            kept.append(row)
    assert len({r["utt"] for r in kept}) == len(kept), "duplicate utt id"

    per = {"train": kept, "dev": [], "test": []}

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "csv": str(args.csv),
        "dropped": {k: v for k, v in dropped.items() if k != "kept"},
        "n_utt": len(kept),
        "splits": {},
        "leakage": {},
    }
    for name in NAMES:
        dest = args.out / name
        dest.mkdir(parents=True, exist_ok=True)
        sel = sorted(per[name], key=lambda r: r["utt"])
        (dest / "wav.scp").write_text("".join(f"{r['utt']} {r['wav']}\n" for r in sel))
        (dest / "text").write_text("".join(f"{r['utt']} {r['text']}\n" for r in sel))
        (dest / "utt2spk").write_text("".join(f"{r['utt']} {r['case']}\n" for r in sel))
        spk = collections.defaultdict(list)
        for r in sel:
            spk[r["case"]].append(r["utt"])
        (dest / "spk2utt").write_text("".join(f"{k} {' '.join(v)}\n" for k, v in sorted(spk.items())))
        (dest / "instruct").write_text("".join(f"{r['utt']} {INSTRUCT}\n" for r in sel))
        report["splits"][name] = {
            "n_utt": len(sel),
            "n_doctor": len({r["case"] for r in sel}),
            "hours": round(sum(r["dur"] for r in sel) / 3600, 3),
            "doctors": sorted({r["case"] for r in sel}),
        }
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        report["leakage"][f"{a}-{b}"] = len({r["case"] for r in per[a]} & {r["case"] for r in per[b]})
    (args.out / "split_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for name in NAMES:
        s = report["splits"][name]
        print(f"{name:6s} {s['n_utt']:4d} utt {s['hours']:6.3f} h  {s['doctors']}")
    print(f"dropped {report['dropped']} | leakage {report['leakage']}")


if __name__ == "__main__":
    main()
