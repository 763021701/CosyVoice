#!/usr/bin/env python3
"""Second instrument for the pathology domain: how close are the synthesized speech tokens to the real ones?

The ASR round-trip has no resolving power on this domain -- Qwen3-ASR transcribes the real dictation at 86%
unit error ("Foveolar hyperplasia is noted" -> "除非呢个place是不footstop"), so a synthesized arm scoring 34%
is not better than the doctor, it is just easier for the ASR to decode. What we can still measure without a
human listener is the token sequence the same tokenizer assigns to the two audios:

  dist(synth_i, real_i)  -- same text, same tokenizer: does the generated audio sit near the real one?
  dist(real_i, real_j)   -- same doctor, different sentence: the calibration for "not close at all"

Both are reported. Without the second number the first one is uninterpretable.

Also reports an alignment-free histogram cosine, because duration mismatch alone inflates edit distance, and
the length ratio, since a stretched clip is a real failure mode of the fine-tuned LLM.

Requires tools/extract_speech_token.py to have been run over each arm's synthesized wavs (it wants a Kaldi
dir, so synth/<arm>/wav.scp maps utt -> the generated file).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import torch


def read_scp(path: Path) -> dict:
    return {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
            for l in path.read_text().splitlines()}


def as_list(x) -> list:
    """utt2speech_token.pt holds tensors in some tools' output and plain lists in others."""
    return x.tolist() if hasattr(x, 'tolist') else [int(v) for v in x]


def lev(a: list[int], b: list[int], cap: int = 1500) -> int:
    """Banded Levenshtein. Full DP on 900x700 token sequences is 600k cells per pair in Python, and there
    are hundreds of pairs; the band is the length difference plus 15% slack, which no realignment exceeds."""
    a, b = a[:cap], b[:cap]
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return max(n, m)
    band = max(abs(n - m) + int(0.15 * max(n, m)), 20)
    INF = float('inf')
    prev = [INF] * (m + 1)
    prev[0] = 0
    for i in range(1, n + 1):
        cur = [INF] * (m + 1)
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        cur[j0 - 1] = INF
        left = INF
        for j in range(j0, j1 + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            # row recurrence: insertion uses the current row's left neighbour, deletion the previous row,
            # substitution the previous row's diagonal (held in `left`)
            up = prev[j] + 1
            lo = cur[j - 1] + 1 if j > j0 else INF
            di = left + cost
            cur[j] = min(up, lo, di)
            left = prev[j]
        prev = cur
    return int(prev[m]) if prev[m] != INF else band + max(n, m)


def hist_cos(a: list[int], b: list[int]) -> float:
    ca, cb = collections.Counter(a), collections.Counter(b)
    keys = set(ca) | set(cb)
    dot = sum(ca[k] * cb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else float('nan')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out-root', type=Path, required=True, help='dir holding <arm>/utt2speech_token.pt')
    ap.add_argument('--arms', nargs='+', required=True)
    args = ap.parse_args()

    real = torch.load(args.data_root / args.split / 'utt2speech_token.pt', map_location='cpu', weights_only=True)
    utt2case = {k: v for k, v in
                ((l.split(maxsplit=1)[0], l.split(maxsplit=1)[1])
                 for l in (args.data_root / args.split / 'utt2spk').read_text().splitlines())}
    by_case = collections.defaultdict(list)
    for u, c in utt2case.items():
        by_case[c].append(u)

    # calibration: real vs another real utterance of the same case (same doctor, different sentence)
    base_d, base_c = [], []
    for c, us in by_case.items():
        for i, u in enumerate(us):
            v = us[(i + 1) % len(us)]
            if v == u:
                continue
            a, b = as_list(real[u]), as_list(real[v])
            base_d.append(lev(a, b) / max(len(a), len(b), 1))
            base_c.append(hist_cos(a, b))
    print(f'{"comparison":26s} {"n":>4} {"edit/ref":>9} {"hist-cos":>9} {"len ratio":>10}')
    print(f'{"real vs real (same case)":26s} {len(base_d):4d} {sum(base_d)/len(base_d):9.3f} '
          f'{sum(base_c)/len(base_c):9.3f} {"":>10}')

    rows = {}
    for arm in args.arms:
        f = args.out_root / arm / 'utt2speech_token.pt'
        if not f.is_file():
            print(f'{arm:26s} -- no {f}, run tools/extract_speech_token.py over the synth dir')
            continue
        tok = torch.load(f, map_location='cpu', weights_only=True)
        d, cs, lr = [], [], []
        empty = 0
        for u, t in tok.items():
            if u not in real:
                continue
            a, b = as_list(real[u]), as_list(t)
            if not a or not b:  # a zero-token generation: a failure, not a distance of 0
                empty += 1
                continue
            d.append(lev(a, b) / max(len(a), len(b), 1))
            cs.append(hist_cos(a, b))
            lr.append(len(b) / len(a))
        rows[arm] = {'n': len(d), 'edit': sum(d)/len(d), 'hist_cos': sum(cs)/len(cs),
                     'len_ratio': sum(lr)/len(lr), 'zero_token': empty}
        v = rows[arm]
        print(f'{arm:26s} {v["n"]:4d} {v["edit"]:9.3f} {v["hist_cos"]:9.3f} {v["len_ratio"]:10.3f}'
              f'   zero-token {empty}')
    (args.out_root / 'token_distance.json').write_text(
        json.dumps({'baseline_real_vs_real': {'n': len(base_d), 'edit': sum(base_d)/len(base_d),
                                              'hist_cos': sum(base_c)/len(base_c)},
                    'arms': rows}, indent=1) + '\n')
    print(f'-> {args.out_root / "token_distance.json"}')


if __name__ == '__main__':
    main()
