#!/usr/bin/env python3
"""Drop segments with a long internal pause, trim long leading/trailing silence, copy -- never edit -- the audio.

Why: the fine-tuned LLM's failure mode was emitting near-constant low-id speech tokens (243/270), which the
flow renders as digital silence. Those ids are exactly what the tokenizer assigns to long quiet stretches, and
the pathology cuts are full of them -- a doctor stops for several seconds mid-sentence. Training on that teaches
"long silence is a thing to generate", and the SwitchLingua data (median 4.95s, spoken in one breath) cannot
outvote it.

Rules, per the experiment agreed:
  reject   a segment whose longest *internal* low-energy run >= --max-gap (default 1.0 s)
  trim     leading/trailing silence down to --lead / --trail margins (default 0.10 / 0.30 s)
  write    the kept audio as a new file under <out>/<split>/wav/, original files untouched

Threshold is per-file: silence = frames below (95th-percentile frame energy - 25 dB), which is the same
relative-energy idea the corpus aligner uses (it stored noise_db/speech_db/threshold_db per recording), and
robust to the fact that MossFormer2 output has a near -80 dBFS noise floor while the 16 kHz recordings do not.

--dry-run reports the distributions and writes nothing.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

FR = 0.01  # 10 ms analysis frame


def speech_runs(x: np.ndarray, sr: int) -> tuple[list[tuple[float, float]], float, int]:
    n = int(len(x) * 1000 / sr) // 10
    hop = int(sr * FR)
    db = 10 * np.log10((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    ref = float(np.percentile(db, 95))
    thr = max(ref - 25.0, -75.0)
    voiced = db > thr
    runs, st = [], None
    for i, v in enumerate(voiced):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append((st * FR, i * FR))
            st = None
    if st is not None:
        runs.append((st * FR, n * FR))
    merged = []  # join runs separated by < 0.1 s so a fricative tail is not read as a pause
    for a, b in runs:
        if merged and a - merged[-1][1] < 0.1:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged, ref, thr


def analyze(path: str, dur: float) -> dict:
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    runs, ref, thr = speech_runs(np.asarray(x), sr)
    if not runs:
        return dict(dur=dur, lead=dur, trail=0.0, gap=dur, nspeech=0, ref=ref, thr=thr)
    gap = max((runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)), default=0.0)
    return dict(dur=dur, lead=round(runs[0][0], 3), trail=round(dur - runs[-1][1], 3),
                gap=round(gap, 3), nspeech=len(runs), ref=ref, thr=thr)


def trim(path: str, dest: str, a: dict, lead: float, trail: float) -> tuple[float, float]:
    """Cut to the speech span plus margins. Returns the new start/end in seconds."""
    x, sr = sf.read(path)
    st = max(0.0, a['lead'] - lead)
    en = min(a['dur'], a['dur'] - a['trail'] + trail)
    if en - st < 0.5:  # the margins ate the clip; keep everything we have
        st, en = 0.0, a['dur']
    i0, i1 = int(st * sr), int(en * sr)
    sf.write(dest, x[i0:i1], sr, subtype='PCM_16')
    return st, en


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=Path('data_mix'), help='Kaldi dirs to read (split/text/wav.scp)')
    ap.add_argument('--out', type=Path, default=Path('data_mix_clean'))
    ap.add_argument('--max-gap', type=float, default=1.0, help='reject if an internal pause is at least this long')
    ap.add_argument('--lead', type=float, default=0.10, help='leading silence to keep after trimming')
    ap.add_argument('--trail', type=float, default=0.30, help='trailing silence to keep after trimming')
    ap.add_argument('--min-dur', type=float, default=1.0, help='drop trimmed clips shorter than this')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    report = {'params': {k: getattr(args, k) for k in ('max_gap', 'lead', 'trail', 'min_dur')}, 'splits': {}}
    for split_dir in sorted(p for p in args.src.iterdir() if (p / 'wav.scp').is_file()):
        split = split_dir.name
        wav = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
               for l in (split_dir / 'wav.scp').read_text().splitlines()}
        text = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
                for l in (split_dir / 'text').read_text().splitlines()}
        spk = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
               for l in (split_dir / 'utt2spk').read_text().splitlines()}
        instruct_path = split_dir / 'instruct'
        inst = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
                for l in instruct_path.read_text().splitlines()} if instruct_path.is_file() else {}

        kept, dropped, stats = [], [], []
        out_wav = args.out / split / 'wav'
        if not args.dry_run:
            out_wav.mkdir(parents=True, exist_ok=True)
        for u, w in wav.items():
            info = sf.info(w)
            a = analyze(w, info.frames / info.samplerate)
            a['utt'] = u
            if a['nspeech'] == 0:
                dropped.append({**a, 'why': 'no_speech_at_all'})
                continue
            if a['gap'] >= args.max_gap:
                dropped.append({**a, 'why': f'internal_gap>={args.max_gap}'})
                continue
            if not args.dry_run:
                st, en = trim(w, str(out_wav / f'{u}.wav'), a, args.lead, args.trail)
                a['new_dur'] = round(en - st, 3)
            else:
                a['new_dur'] = round(a['dur'] - max(0, a['lead'] - args.lead) - max(0, a['trail'] - args.trail), 3)
            if a['new_dur'] < args.min_dur:
                dropped.append({**a, 'why': 'too_short_after_trim'})
                continue
            kept.append({**a, 'path': str((out_wav / f'{u}.wav').resolve()) if not args.dry_run else w})
        if not kept:
            print(f'{split}: nothing survives')
            continue
        q = lambda vals, p=0.9: sorted(vals)[int(p * (len(vals) - 1))]
        print(f'{split}: kept {len(kept)} / {len(wav)}  dropped {len(dropped)} '
              f'{dict(collections.Counter(d["why"] for d in dropped))}')
        print(f'   kept 中位时长 {np.median([k["new_dur"] for k in kept]):.2f}s | 内部最大停顿 '
              f'p50 {q([k["gap"] for k in kept]):.2f}s p90 {q([k["gap"] for k in kept]):.2f}s max '
              f'{max(k["gap"] for k in kept):.2f}s | 裁掉的前静音 p50 {q([k["lead"] for k in kept]):.2f}s '
              f'max {max(k["lead"] for k in kept):.2f}s | 后静音 p50 {q([k["trail"] for k in kept]):.2f}s '
              f'max {max(k["trail"] for k in kept):.2f}s')
        if not args.dry_run:
            d = args.out / split
            (d / 'wav.scp').write_text(''.join(f"{k['utt']} {k['path']}\n" for k in kept))
            (d / 'text').write_text(''.join(f"{k['utt']} {text[k['utt']]}\n" for k in kept))
            (d / 'utt2spk').write_text(''.join(f"{k['utt']} {spk[k['utt']]}\n" for k in kept))
            by = collections.defaultdict(list)
            for k in kept:
                by[spk[k['utt']]].append(k['utt'])
            (d / 'spk2utt').write_text(''.join(f'{s} {" ".join(v)}\n' for s, v in sorted(by.items())))
            if inst:
                (d / 'instruct').write_text(''.join(f"{k['utt']} {inst[k['utt']]}\n" for k in kept))
            (d / 'silence_stats.jsonl').write_text(
                ''.join(json.dumps({**k, 'src_wav': wav[k['utt']]}, ensure_ascii=False) + '\n' for k in kept))
        report['splits'][split] = {
            'n_kept': len(kept), 'n_dropped': len(dropped),
            'drop_reasons': dict(collections.Counter(x['why'] for x in dropped)),
            'per_case_kept': dict(collections.Counter(spk[k['utt']] for k in kept)),
            'per_case_dropped': dict(collections.Counter(spk[x['utt']] for x in dropped)),
            'kept_dur_median': float(np.median([k['new_dur'] for k in kept])),
            'dropped_detail': [dict(utt=x['utt'], why=x['why'], gap=x['gap'], dur=x['dur']) for x in dropped],
        }
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'clean_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
        print(f'-> {args.out}')


if __name__ == '__main__':
    main()
