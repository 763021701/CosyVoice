#!/usr/bin/env python3
"""Tier-2 scoring: Qwen3-ASR round-trip on synthesized test speech, split by language span.

Run in the qwen3_asr_ft env (transformers 4.57), not the cosyvoice env: the vendored qwen_asr modeling
code uses check_model_inputs(func=...) and fails to import under transformers 5.

    /root/autodl-tmp/miniconda3/envs/qwen3_asr_ft/bin/python local/asr_score.py ...

Metrics are computed on code-switching units, not bytes or words alone: one CJK character is a unit, one
Latin run is a unit, one digit run is a unit, everything else (punctuation) is dropped. Per-unit outcome
(correct / substituted / deleted / inserted) comes from the edit-distance alignment backtrace, which is
what lets EN_ERR and ZH_ERR be reported separately instead of one blended WER.

The real test audio is transcribed through the identical path and reported as a floor. This is not
optional decoration: Qwen3-ASR may emit simplified characters for our traditional-Cantonese references, or
expand numbers differently, and those are ASR biases rather than TTS errors. Without the floor a CJK error
rate cannot be read as "the synthesised speech was wrong"; with it, the meaningful quantity is the gap
between a synthesized arm and real audio on the same ASR and the same scorer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/root/autodl-tmp/workspace/project/dev/Qwen3-ASR-main')))

import torch
from qwen_asr import Qwen3ASRModel

UNIT = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+|[㐀-䶿一-鿿]")


def units(text: str) -> list[str]:
    return [t.lower() for t in UNIT.findall(text)]


def kind(u: str) -> str:
    if u.isascii() and any(c.isalpha() for c in u):
        return 'en'
    if u.isdigit():
        return 'num'
    return 'zh'


def align(ref: list[str], hyp: list[str]):
    """Levenshtein with backtrace -> per-ref-unit outcome plus insertion count."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            out.append(('sub' if ref[i - 1] != hyp[j - 1] else 'ok', ref[i - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append(('del', ref[i - 1]))
            i -= 1
        else:
            out.append(('ins', None))
            j -= 1
    return out[::-1]


def score(pairs):
    """pairs: list of (ref_text, hyp_text). 'all' is sub+del over ref units; 'overall' adds insertions."""
    agg = {k: {'ref': 0, 'err': 0} for k in ('all', 'en', 'zh', 'num')}
    ins = tot_err = n_utt = 0
    for ref_text, hyp_text in pairs:
        r, h = units(ref_text), units(hyp_text)
        if not r:
            continue
        n_utt += 1
        for op, u in align(r, h):
            if op == 'ins':
                ins += 1
                tot_err += 1
                continue
            g = kind(u)
            for bucket in ('all', g):
                agg[bucket]['ref'] += 1
                if op != 'ok':
                    agg[bucket]['err'] += 1
            tot_err += (op != 'ok')
    out = {b: (v['err'] / v['ref'] if v['ref'] else float('nan')) for b, v in agg.items()}
    # ref counts travel with the rates: a bucket with 1 unit in it reads as 0% or 100% and is meaningless
    out.update({f'n_{b}': v['ref'] for b, v in agg.items()})
    out['n_utt'] = n_utt
    out['overall'] = tot_err / max(agg['all']['ref'], 1)
    out['ins_per_ref'] = ins / max(agg['all']['ref'], 1)
    return out


def transcribe(model, paths, batch=32):
    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = model.transcribe(audio=chunk, language=None)
        out.update({p: r.text for p, r in zip(chunk, res)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--asr-model', default='/root/autodl-tmp/models/Qwen3-ASR-0.6B')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, required=True, help='dir with <arm>.wav/ per checkpoint')
    ap.add_argument('--arms', nargs='*', default=[],
                    help='checkpoint names under out-root; empty means measure the real-audio floor only')
    ap.add_argument('--split', default='test')
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    text = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
            for l in (args.data_root / args.split / 'text').read_text().splitlines()}
    real = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
            for l in (args.data_root / args.split / 'wav.scp').read_text().splitlines()}
    utts = sorted(text)

    # Score the intersection only. An arm that dropped utterances to a degenerate sample would otherwise be
    # compared against a different set of sentences than its partner.
    if args.arms:
        have = {a: {u for u in utts if (args.out_root / a / f'{u}.wav').is_file()} for a in args.arms}
        keep = set.intersection(*have.values()) if have else set()
        print(f'per-arm wav counts {[(a, len(v)) for a, v in have.items()]}, '
              f'scoring the intersection: {len(keep)} utt')
        utts = sorted(keep)

    model = Qwen3ASRModel.from_pretrained(args.asr_model, dtype=torch.bfloat16, device_map='cuda:0',
                                          max_inference_batch_size=args.batch, max_new_tokens=256)

    results, hyp_store = {}, {}
    arms = [('real', [real[u] for u in utts])] + \
           [(a, [str(args.out_root / a / f'{u}.wav') for u in utts]) for a in args.arms]
    for name, paths in arms:
        hyp = transcribe(model, paths, args.batch)
        hyp_store[name] = hyp
        pairs = [(text[u], hyp[paths[i]]) for i, u in enumerate(utts)]
        s = score(pairs)
        dur = None
        mf = args.out_root / f'{name}.manifest.json'
        if mf.is_file():
            # restricted to the scored intersection so the duration number describes the same samples
            m = [x for x in json.loads(mf.read_text()) if x['utt'] in set(utts)]
            dur = sum(x['synth_sec'] for x in m) / sum(x['real_sec'] for x in m)
        results[name] = {'err': {k: round(s[k], 4) for k in ('all', 'en', 'zh', 'num')},
                         'n_ref': {k: s[f'n_{k}'] for k in ('all', 'en', 'zh', 'num')},
                         'overall': round(s['overall'], 4), 'ins_per_ref': round(s['ins_per_ref'], 4),
                         'duration_ratio': round(dur, 4) if dur else None}
        print(f"{name:10s} overall {s['overall'] * 100:6.2f}%   "
              f"en {s['en'] * 100:6.2f}% (n={s['n_en']})   "
              f"zh {s['zh'] * 100:6.2f}% (n={s['n_zh']})   "
              f"num {s['num'] * 100:5.1f}% (n={s['n_num']})   "
              f"ins/ref {s['ins_per_ref'] * 100:5.2f}%"
              f"{'' if dur is None else f'   dur x{dur:.2f}'}")

    floor = results['real']
    print(f"\ngap over the real-audio floor ({floor['overall'] * 100:.2f}%):")
    for a in args.arms:
        print(f"  {a:10s} overall {(results[a]['overall'] - floor['overall']) * 100:+.2f}pp   "
              f"en {(results[a]['err']['en'] - floor['err']['en']) * 100:+.2f}pp   "
              f"zh {(results[a]['err']['zh'] - floor['err']['zh']) * 100:+.2f}pp")

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / 'asr_results.json').write_text(
        json.dumps({'metrics': results, 'hyp': hyp_store}, indent=1, ensure_ascii=False) + '\n')
    print(f"\n-> {args.out_root / 'asr_results.json'}")


if __name__ == '__main__':
    main()
