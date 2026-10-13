#!/usr/bin/env python3
"""Recover approximate speaker groups from the extracted campplus embeddings.

SwitchLingua ships no speaker id, but tier-2 synthesis needs a prompt clip that is *not* the utterance
being synthesized (frontend_zero_shot feeds the prompt's speech tokens into the LLM, so self-clone would
hand the model its own answer). Same-speaker-other-clip is the design used elsewhere; to get it here we
cluster all 7272 campplus embeddings and pair within a cluster.

Threshold is chosen on two independent signals, not on the clustering's own silhouette:
  * gender purity -- campplus should never put a male and a female clip in one cluster. This catches gross
    under-clustering (threshold too loose). It cannot see over-splitting, hence the second signal.
  * same-item co-clustering -- the '{item}_{sentence}' file names mean sentences of one story were read in
    one session, so they are very likely one speaker. A good threshold keeps items together.
The README says 174 bilingual speakers overall, so the cluster count is a third, weak sanity read.

Writes data/speakers.json {utt: cluster} plus the scan table used to pick the threshold.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

SPLITS = ("train", "dev", "test")


def load_embeddings(data_root: Path):
    utts, vecs = [], []
    for s in SPLITS:
        table = torch.load(data_root / s / "utt2embedding.pt", map_location="cpu", weights_only=True)
        for utt in sorted(table):
            utts.append(utt)
            vecs.append(table[utt])
    x = np.asarray(vecs, dtype=np.float32)
    return utts, x / np.linalg.norm(x, axis=1, keepdims=True)


def load_meta(csv_path: Path, wav_dir: Path):
    import csv
    csv.field_size_limit(10 ** 9)
    have = {p.stem for p in (wav_dir).glob("*.wav")}
    gender, item = {}, {}
    for r in csv.DictReader((csv_path).open(encoding="utf-8-sig", newline="")):
        stem = r["file_name"].rsplit(".", 1)[0]
        if stem in have:
            gender[stem] = r["gender"]
            item[stem] = stem.split("_")[0]
    return gender, item


def gender_purity(labels, gender, utts):
    by_c = defaultdict(list)
    for u, l in zip(utts, labels):
        by_c[l].append(gender[u])
    return sum(max(Counter(v).values()) for v in by_c.values()) / len(utts)


def item_coherence(labels, item, utts):
    """Fraction of same-item pairs that landed in the same cluster."""
    by_item = defaultdict(list)
    for u, l in zip(utts, labels):
        by_item[item[u]].append(l)
    num = den = 0
    for ls in by_item.values():
        n = len(ls)
        if n < 2:
            continue
        c = Counter(ls)
        num += sum(k * (k - 1) for k in c.values())
        den += n * (n - 1)
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--pick", type=float, required=True, help="cosine distance threshold to keep")
    args = ap.parse_args()

    utts, x = load_embeddings(args.data_root)
    gender, item = load_meta(args.csv, args.wav_dir)
    print(f"{len(utts)} utt, dim {x.shape[1]}")

    print(f"\n{'dist<':>6} {'clusters':>9} {'gender_purity':>14} {'item_coherence':>15} {'max_size':>9}")
    chosen = None
    for t in [float(v) for v in args.thresholds.split(",")]:
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric="cosine",
                                      linkage="average").fit_predict(x)
        sizes = Counter(lab.tolist())
        print(f"{t:>6.2f} {len(sizes):>9} {gender_purity(lab, gender, utts):>14.4f} "
              f"{item_coherence(lab, item, utts):>15.4f} {max(sizes.values()):>9}")
        if abs(t - args.pick) < 1e-9:
            chosen = lab
    if chosen is None:
        raise SystemExit(f"--pick {args.pick} not in the scanned thresholds")

    out = args.data_root / "speakers.json"
    out.write_text(json.dumps({u: int(c) for u, c in zip(utts, chosen)}, ensure_ascii=False, indent=0))
    sizes = Counter(chosen.tolist())
    print(f"\nthreshold {args.pick}: {len(sizes)} clusters -> {out}")
    print(f"cluster size: 中位 {int(np.median(list(sizes.values())))}, max {max(sizes.values())}, "
          f"单例 {sum(1 for v in sizes.values() if v == 1)}")


if __name__ == "__main__":
    main()
