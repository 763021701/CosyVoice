#!/usr/bin/env python3
"""Prepare a text-only generation corpus as four synthesis cells: {A,B} corpora x {slref,pathref} prompt libraries.

What this answers: the same LLM (`data_mix_clean/ft_ep1`) and the same text, with only the zero-shot voice
reference swapped between the SwitchLingua pool and the pathology pool. Everything downstream -- synthesis,
ASR round-trip, punctuation realization, timbre following -- then differs by one variable.

Decisions baked in here (plan: ~/.claude/plans/streamed-floating-cocoa.md):

* text: normalize *word forms* to the training convention (RULES imported from prepare_mix_data, never
  re-typed), but never ADD a punctuation word to a bare sentence-final `.` -- real dictation only says it at
  21.0% of sentence ends, so inventing it everywhere would synthesize a style no doctor uses.
* prompt selection: round-robin over the whole library, in *dealt* order. Sorting the library lexicographically
  would hand adjacent targets the same reader (an SL item has up to 16 clips, a pathology case up to 42);
  dealing emits group r of every speaker before group r+1, so the first 1,701 / 28 slots are all distinct.
* libraries: SL = all its clips (the sealed test is prompt material only -- no target text comes from it, so
  nothing measured is contaminated). Pathology = raw untrimmed audio straight from the two source jsonl (NOT
  data_mix_clean, whose edges were cut, and NOT data_mix's 559, which cannot hold the 660 asked for). Kept:
  clips whose transcript marks a hesitation with `...` -- that marker means the reader paused or restarted, so
  text and audio DO align; only >30 s (hard assert in cosyvoice/cli/frontend.py:97 would kill the worker),
  <1 s, and the duplicated case-13488 rows are excluded.

The corpora are a moving target (both files were edited while this plan was being written), so the md5, mtime
and row count of every input go into plan.json and the normalized text is snapshotted: pilot and full run read
the same snapshot rather than re-opening the source files.

    python -I local/make_synth_cells.py --dry-run
    python -I local/make_synth_cells.py --pilot-per-cell 60
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_mix_data import normalize, rows_from_jsonl, unusable  # noqa: E402

TS = Path('/root/autodl-tmp/workspace/dataset/Test_Samples')
REPO = HERE.parents[2]     # HERE is .../examples/switchlingua-yue/local, so the repo root is two levels up
SPLITS = ('train', 'dev', 'test')
MAX_PROMPT_SEC = 30.0     # frontend.py:97 asserts on prompt length
MIN_PROMPT_SEC = 1.0      # too little voice to condition on
SPACED_ID = re.compile(r'[〇一二三四五六七八九十两] [〇一二三四五六七八九十两]')


def read_scp(path: Path) -> dict:
    return {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1] for l in path.read_text().splitlines()}


def duration(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return info.frames / info.samplerate


def deal(rows: list[dict], speaker: str) -> list[dict]:
    """Round-robin across speaker groups so adjacent slots are different readers."""
    groups = {}
    for r in rows:
        groups.setdefault(r[speaker], []).append(r)
    for v in groups.values():
        v.sort(key=lambda r: r['utt'])
    order = []
    for i in range(max(len(v) for v in groups.values())):
        for g in sorted(groups):
            if len(groups[g]) > i:
                order.append(groups[g][i])
    return order


def sl_library(root: Path, workers: int) -> tuple[list[dict], dict, float]:
    rows = []
    for s in SPLITS:
        wav, text = read_scp(root / s / 'wav.scp'), read_scp(root / s / 'text')
        rows += [{'utt': u, 'split': s, 'wav': wav[u], 'text': text[u], 'item': u.split('_')[0]}
                 for u in sorted(wav)]
    with cf.ThreadPoolExecutor(workers) as ex:
        for r, d in zip(rows, ex.map(lambda r: duration(r['wav']), rows)):
            r['dur'] = round(d, 3)
    reasons = [(None if Path(r['wav']).is_file() else 'missing_wav')
               or ('too_long' if r['dur'] > MAX_PROMPT_SEC else None)
               or ('too_short' if r['dur'] < MIN_PROMPT_SEC else None)
               or ('empty_text' if not r['text'].strip() else None) for r in rows]
    kept, counts = [], {}
    for r, why in zip(rows, reasons):
        if why:
            counts[why] = counts.get(why, 0) + 1
        else:
            kept.append(r)
    return kept, counts, sum(r['dur'] for r in kept) / 3600


def path_library(seg: Path, meta: Path, exclude_hesitation: bool) -> tuple[list[dict], dict, float, int]:
    """Library rows come from the same reader as training (rows_from_jsonl), so prompt text is normalized
    exactly like the transcripts the model was trained on. Only the membership test differs."""
    prompt_audit: list[dict] = []
    n_src = len(seg.read_text().splitlines()) + len(meta.read_text().splitlines())
    rows = rows_from_jsonl(seg, 'from_denoised', prompt_audit) + rows_from_jsonl(meta, 'metadata', prompt_audit)
    # rows_from_jsonl already drops the duplicated case (DUP_CASE) before returning; make that visible in the
    # exclusion table rather than leaving it as arithmetic the reader has to do in their head
    counts = {'duplicated_case': n_src - len(rows)}
    kept = []
    for r in rows:
        why = unusable(r)
        if why == 'gap_in_transcript' and not exclude_hesitation:
            why = None          # `...` marks a hesitation, not a transcription hole: audio aligns
        if why:
            counts[why] = counts.get(why, 0) + 1
        else:
            kept.append(r)
    return kept, counts, sum(r['dur'] for r in kept) / 3600, len(prompt_audit)


def load_corpus(path: Path, tag: str, stratum: str, audit: list) -> list[dict]:
    """Snapshot a text jsonl into {utt, raw, text(norm), stratum, meta} rows. File order == utt order because
    both ids are zero-padded, which is what keeps the rotation index aligned with synth_test.py's sorted walk."""
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        r = json.loads(line)
        if tag == 'A':
            utt = f'A{i:05d}'
            meta = {k: r.get(k) for k in ('category_id', 'phrase', 'priority', 'variant')}
        else:
            utt = f'B{int(r["id"]):05d}'
            meta = {k: r.get(k) for k in ('origin', 'mode', 'section', 'length', 'target_term', 'source')}
        text, edits = normalize(r['text'])
        if edits:
            audit.append({'utt': utt, 'src': tag, 'edits': edits, 'before': r['text'], 'after': text})
        if text.strip():
            out.append({'utt': utt, 'raw': r['text'], 'text': text,
                        'stratum': meta.get(stratum) or 'none', 'meta': meta})
    return out


def pick(rows: list[dict], n: int, wants: list[tuple]) -> list[dict]:
    """Deterministic stratified sample: forced coverage first (wants = (predicate, count)), then one row per
    stratum at a time. No RNG -- the sample is reproducible from the input files alone."""
    taken, out = set(), []
    for pred, k in wants:
        got = 0
        for r in rows:
            if got >= k:
                break
            if r['utt'] not in taken and pred(r):
                out.append(r)
                taken.add(r['utt'])
                got += 1
    by = {}
    for r in rows:
        if r['utt'] not in taken:
            by.setdefault(r['stratum'], []).append(r)
    while len(out) < n and any(by.values()):
        for g in sorted(by):
            if by[g] and len(out) < n:
                out.append(by[g].pop(0))
    return sorted(out, key=lambda r: r['utt'])


def census(tag: str, rows: list[dict]) -> dict:
    t = [r['text'] for r in rows]
    return {'rows': len(rows),
            'sentence_ends': sum(x.count('.') for x in t),
            'spoken_punct_words': sum(x.count('Full stop') + x.count('Period') + x.count('comma') for x in t),
            'with_dash_word': sum('dash' in x for x in t),
            'with_c_m': sum(' c m' in x for x in t),
            'with_mm': sum('mm' in x for x in t),
            'with_spaced_id': sum(bool(SPACED_ID.search(x)) for x in t),
            'chars_median': sorted(len(x) for x in t)[len(t) // 2],
            'chars_max': max(len(x) for x in t)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-a', type=Path, default=REPO / 'data/pathology/llm_syn_sentences.jsonl')
    ap.add_argument('--corpus-b', type=Path, default=REPO / 'data/pathology/tts_corpus_final_normalized.jsonl')
    ap.add_argument('--sl-root', type=Path, default=Path('data'), help='SwitchLingua Kaldi root = slref library')
    ap.add_argument('--path-segments', type=Path,
                    default=TS / 'data/processed/segmented/from_denoised/segments.jsonl')
    ap.add_argument('--path-metadata', type=Path,
                    default=TS / 'samples/pathology_0821/output_v6_asr_sync/metadata.jsonl')
    ap.add_argument('--out', type=Path, default=Path('data_synth'))
    ap.add_argument('--pilot-per-cell', type=int, default=60)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--exclude-hesitation-clips', action='store_true',
                    help='drop the pathology clips whose transcript marks a hesitation with ... (they are the '
                         'only library text carrying a character sequence the LLM never saw in training)')
    ap.add_argument('--dry-run', action='store_true', help='print the reports, write nothing')
    args = ap.parse_args()

    audit: list[dict] = []
    corpora = {'A': load_corpus(args.corpus_a, 'A', 'category_id', audit),
               'B': load_corpus(args.corpus_b, 'B', 'mode', audit)}
    sl_rows, sl_counts, sl_hours = sl_library(args.sl_root, args.workers)
    pt_rows, pt_counts, pt_hours, pt_renorm = path_library(args.path_segments, args.path_metadata,
                                                           args.exclude_hesitation_clips)
    libs = {'slref': deal(sl_rows, 'item'), 'pathref': deal(pt_rows, 'case')}

    report = {
        'inputs': {str(p): {'md5': hashlib.md5(p.read_bytes()).hexdigest(), 'mtime': int(p.stat().st_mtime),
                            'bytes': p.stat().st_size, 'rows': len(p.read_text().splitlines())}
                   for p in (args.corpus_a, args.corpus_b, args.path_segments, args.path_metadata)},
        'git': subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True).stdout.strip(),
        'hesitation_clips_excluded': args.exclude_hesitation_clips,
        'libraries': {k: {'kept': len(v),
                          'speakers': len({r['item'] if k == 'slref' else r['case'] for r in v}),
                          'hours': round(sum(r['dur'] for r in v) / 3600, 2),
                          'fingerprint': hashlib.sha256('\n'.join(r['utt'] for r in v).encode()).hexdigest()[:16],
                          'adjacent_same_speaker': sum(
                              1 for a, b in zip(v, v[1:])
                              if (a['item'] if k == 'slref' else a['case']) == (b['item'] if k == 'slref' else b['case']))}
                      for k, v in libs.items()},
        'dropped': {'slref': sl_counts, 'pathref': pt_counts, 'path_prompt_texts_renormalized': pt_renorm},
        'census': {k: census(k, v) for k, v in corpora.items()},
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f'normalized {len(audit)} target texts (word-form rules only; see normalization_audit.jsonl)')

    if args.dry_run:
        print('\n--dry-run: nothing written')
        return

    for tag, rows in corpora.items():
        d = args.out / 'corpus' / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / 'text').write_text('\n'.join(f"{r['utt']} {r['text']}" for r in rows) + '\n')
        (d / 'source_map.jsonl').write_text('\n'.join(
            json.dumps({'utt': r['utt'], 'text_raw': r['raw'], 'text_norm': r['text'],
                        'stratum': r['stratum'], 'raw_record': r['meta']}, ensure_ascii=False)
            for r in rows) + '\n')
    (args.out / 'normalization_audit.jsonl').write_text(
        '\n'.join(json.dumps(a, ensure_ascii=False) for a in audit) + '\n')
    (args.out / 'libraries').mkdir(parents=True, exist_ok=True)
    for name, rows in libs.items():
        (args.out / 'libraries' / f'{name}.library.jsonl').write_text('\n'.join(
            json.dumps(dict(r, pos=i), ensure_ascii=False) for i, r in enumerate(rows)) + '\n')
    (args.out / 'libraries' / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')

    wants = {'A': [(lambda r: r['meta'].get('category_id') == 'spoken_punctuation', 6),
                   (lambda r: 'Full stop' in r['text'], 4),
                   (lambda r: 'Period' in r['text'] and r['meta'].get('category_id') != 'spoken_punctuation', 6),
                   (lambda r: 'comma' in r['text'], 4),
                   (lambda r: ' c m' in r['text'], 6),
                   (lambda r: 'mm' in r['text'], 4),
                   (lambda r: 'dash' in r['text'], 3),
                   (lambda r: bool(SPACED_ID.search(r['text'])), 5),
                   (lambda r: len(r['text']) > 160, 4)],
             'B': [(lambda r: 'Full stop' in r['text'], 2),
                   (lambda r: 'Period' in r['text'], 8),
                   (lambda r: ' c m' in r['text'], 6),
                   (lambda r: 'mm' in r['text'], 5),
                   (lambda r: 'dash' in r['text'], 5),
                   (lambda r: bool(SPACED_ID.search(r['text'])), 3),
                   (lambda r: len(r['text']) > 300, 3)]}
    cells = {}
    for tag, rows in corpora.items():
        pilot = pick(rows, args.pilot_per_cell, wants[tag])
        for lib, librows in libs.items():
            cell = f'{tag}_{lib}'
            pairs = {r['utt']: {'prompt_utt': p['utt'], 'prompt_wav': p['wav'], 'prompt_text': p['text'],
                                'same_item': False, 'cos': None, 'fallback': False, 'library': lib,
                                'prompt_speaker': p['item'] if lib == 'slref' else p['case']}
                     for r, p in zip(rows, (librows[i % len(librows)] for i in range(len(rows))))}
            for name, subset in ((cell, rows), (f'{cell}_pilot', pilot)):
                d = args.out / 'cells' / name
                d.mkdir(parents=True, exist_ok=True)
                (d / 'text').write_text('\n'.join(f"{r['utt']} {r['text']}" for r in subset) + '\n')
                (args.out / 'cells' / f'{name}_prompt_pairs.json').write_text(
                    json.dumps({r['utt']: pairs[r['utt']] for r in subset}, indent=1, ensure_ascii=False) + '\n')
            cells[cell] = {'n': len(rows), 'n_pilot': len(pilot),
                           'n_prompts': len({v['prompt_utt'] for v in pairs.values()}),
                           'n_speakers': len({v['prompt_speaker'] for v in pairs.values()}),
                           'reuse': round(len(rows) / len(librows), 2)}
    plan = dict(report, cells=cells, total_clips=sum(c['n'] for c in cells.values()))
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=1, ensure_ascii=False) + '\n')
    print(f'\n-> {args.out}/cells/  cells={list(cells)}  total {plan["total_clips"]} clips')


if __name__ == '__main__':
    main()
