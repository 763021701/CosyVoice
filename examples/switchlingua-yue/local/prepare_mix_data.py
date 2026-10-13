#!/usr/bin/env python3
"""Build the pathology-domain half of the mixed fine-tune: two dictation sources -> Kaldi dirs, 8:1:1 by case.

Sources (both the user's own clinical audio, Cantonese-English code-switched dictation):
  B  Test_Samples/data/processed/segmented/from_denoised/segments.jsonl     501 seg, human [CUT] tags
  C  Test_Samples/samples/pathology_0821/output_v6_asr_sync/metadata.jsonl  188 seg, reviewed

Why not from_lid_denoised too: it is the same recordings cut by an LID pass instead of by hand (identical
per-case hours and identical word counts), so taking both would put two different texts on one audio.

Three things have to be settled before any of this can train, and each is decided by the label convention
chosen for this run -- `. Full stop.` + Chinese numerals:
 1. Overlap. case 13488 is in both sources (~5 min of the same recording). B keeps it: its cuts are the
    human [CUT] tags, C's are machine-aligned and include 0.7 s fragments.
 2. Divergent surface forms for the same audio event, rewritten mechanically and logged to label_audit.jsonl:
      "m m"  -> "mm"          from_denoised splits the unit, metadata does not
      "period" -> ". Period." keep the spoken word (that is what the doctor said), unify the symbol pattern
      "polypoid comma soft" -> "polypoid, comma, soft"   metadata wraps the comma-word in real commas
    Mixed forms are exactly the ambiguous-mapping failure measured last round.
 3. Segments whose text contains "..." anywhere -- mid-sentence as well as at the end -- are transcripts with
    a hole in them: the audio has words the text does not. They go, with dur < 1 s (no usable tokens) and > 30 s.

Split is disjoint by *case*: segments of one report share a doctor's voice and phrasing, so a per-segment
random split would leak. Cases then go to whichever split is furthest below its hour target, seeded.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

TS = Path('/root/autodl-tmp/workspace/dataset/Test_Samples')
INSTRUCT = 'You are a helpful assistant.<|endofprompt|>'
DUP_CASE = '13488'  # in both sources; kept from B only
NAMES = ('train', 'dev', 'test')
PUNCT_WORD = re.compile(r'Full stop|Period|comma')

# (pattern, replacement, why). Each pattern is idempotent -- it cannot match text already in target form.
RULES = [
    (re.compile(r'(?<![A-Za-z])m m(?![A-Za-z])'), 'mm', 'unit spelling: mm, not "m m"'),
    (re.compile(r'(?<![.] )\s+period(?=\s+[A-Z<]|$)'), '. Period.', 'punctuation word: unify symbol pattern, keep the word'),
    (re.compile(r'(?<![,.])\s+comma\b'), ', comma,', 'punctuation word: wrap the comma-word in real commas'),
]


def normalize(text: str):
    out, edits = text, []
    for pat, rep, why in RULES:
        new = pat.sub(rep, out)
        if new != out:
            edits.append({'rule': why, 'n': len(pat.findall(out))})
            out = new
    return re.sub(r'\s+', ' ', out).strip(), edits


def rows_from_jsonl(path: Path, src: str, audit: list) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if src == 'from_denoised':
            case, wav, utt = Path(r['source_audio']).stem, str(TS / r['audio']), f"B{Path(r['audio']).stem}"
        else:
            case = Path(r['audio_path']).parent.name
            wav, utt = r['audio_path'], f"C{case}_{Path(r['audio_path']).stem}"
        if src == 'metadata' and case == DUP_CASE:
            continue
        text, edits = normalize(r['text'])
        if edits:
            audit.append({'utt': utt, 'src': src, 'edits': edits, 'after': text})
        out.append({'utt': utt, 'case': case, 'wav': wav, 'text': text, 'dur': float(r['duration']),
                    'punct': len(PUNCT_WORD.findall(text)), 'src': src})
    return out


def unusable(r) -> str | None:
    if not r['text']:
        return 'empty_text'
    if '\n' in r['text']:
        return 'multiline_text'
    if '...' in r['text']:
        return 'gap_in_transcript'
    if r['dur'] < 1.0:
        return 'too_short'
    if r['dur'] > 30.0:
        return 'too_long'
    if not Path(r['wav']).is_file():
        return 'missing_wav'
    return None


def assign_cases(rows, ratios, seed) -> dict:
    """case -> split index: disjoint by case, and every split keeps a share of the punctuation-word doctors.

    A single hour-greedy pass gave dev zero `Full stop`/`Period`/`comma` utterances, because the two longest
    reports happen to be from doctors who never speak their punctuation -- which would leave the CV curve
    blind to exactly the behaviour this run is about. So the cases that do use those words are distributed
    first, against the same 8:1:1 shares, then the rest fills the remaining hours.
    """
    hours = collections.defaultdict(float)
    punct = collections.defaultdict(int)
    for r in rows:
        hours[r['case']] += r['dur']
        punct[r['case']] += r['punct']
    share = [x / sum(ratios) for x in ratios]
    got = [0.0] * len(ratios)
    out: dict = {}
    punct_cases = sorted((c for c in hours if punct[c] > 0), key=lambda c: (-punct[c], -hours[c]))
    # The two loudest punctuation-word doctors go to dev and test, one each, before anything else is placed.
    # Hour-greedy alone kept handing them to train (its share is 8x), which leaves dev blind to the behaviour
    # this run is about -- measured three times: dev at 0, then 1, punct words. Ranked by punct count, not
    # by length, because the longest speaking-punctuation report is not the one that uses it most.
    for slot, c in zip(random.Random(seed).sample([1, 2], 2), punct_cases[:2]):
        out[c] = slot
        got[slot] += hours[c]
    for pool in (punct_cases[2:], sorted(c for c in hours if punct[c] == 0)):
        order = random.Random(seed).sample(pool, len(pool))
        order.sort(key=lambda c: -hours[c])
        for c in order:
            j = min(range(len(ratios)), key=lambda k: got[k] / share[k])
            out[c] = j
            got[j] += hours[c]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ratios', default='8,1,1')
    ap.add_argument('--seed', type=int, default=1986)
    args = ap.parse_args()
    ratios = [float(x) for x in args.ratios.split(',')]

    audit: list = []
    rows = rows_from_jsonl(TS / 'data/processed/segmented/from_denoised/segments.jsonl', 'from_denoised', audit) + \
           rows_from_jsonl(TS / 'samples/pathology_0821/output_v6_asr_sync/metadata.jsonl', 'metadata', audit)
    kept, dropped = [], collections.Counter()
    for r in rows:
        why = unusable(r)
        dropped[why or 'kept'] += 1
        if why is None:
            kept.append(r)
    assert len({r['utt'] for r in kept}) == len(kept), 'duplicate utt id across sources'

    assign = assign_cases(kept, ratios, args.seed)
    per = collections.defaultdict(list)
    for r in kept:
        per[NAMES[assign[r['case']]]].append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {'dropped': {k: v for k, v in dropped.items() if k != 'kept'}, 'n_utt': len(kept),
              'utt_rewritten': len(audit), 'splits': {}, 'leakage': {}}
    for name in NAMES:
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        sel = sorted(per[name], key=lambda r: r['utt'])
        (d / 'wav.scp').write_text(''.join(f"{r['utt']} {r['wav']}\n" for r in sel))
        (d / 'text').write_text(''.join(f"{r['utt']} {r['text']}\n" for r in sel))
        (d / 'utt2spk').write_text(''.join(f"{r['utt']} {r['case']}\n" for r in sel))
        spk = collections.defaultdict(list)
        for r in sel:
            spk[r['case']].append(r['utt'])
        (d / 'spk2utt').write_text(''.join(f'{k} {" ".join(v)}\n' for k, v in sorted(spk.items())))
        (d / 'instruct').write_text(''.join(f"{r['utt']} {INSTRUCT}\n" for r in sel))
        report['splits'][name] = {'n_utt': len(sel), 'n_case': len({r['case'] for r in sel}),
                                  'hours': round(sum(r['dur'] for r in sel) / 3600, 3),
                                  'punct_words': sum(r['punct'] for r in sel),
                                  'by_source': dict(collections.Counter(r['src'] for r in sel)),
                                  'cases': sorted({r['case'] for r in sel})}
    for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        report['leakage'][f'{a}-{b}'] = {
            'case': len({r['case'] for r in per[a]} & {r['case'] for r in per[b]}),
            'utt': len({r['utt'] for r in per[a]} & {r['utt'] for r in per[b]}),
            'text': len({r['text'] for r in per[a]} & {r['text'] for r in per[b]})}
    (args.out / 'split_report_mix.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    (args.out / 'label_audit.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in audit))

    for name in NAMES:
        s = report['splits'][name]
        print(f'{name:6s} {s["n_utt"]:4d} utt {s["n_case"]:3d} case {s["hours"]:6.3f} h  '
              f'punct-words {s["punct_words"]:4d}  {s["by_source"]}')
    print(f'dropped {dict((k, v) for k, v in dropped.items() if k != "kept")} | '
          f'utt rewritten {len(audit)}/{len(kept)} | leakage {report["leakage"]}')


if __name__ == '__main__':
    main()
