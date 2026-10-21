#!/usr/bin/env python3
"""Do the spoken punctuation words get pronounced -- and only when the text asks for them?

Teacher-forced CE cannot answer this. A dropped "Full stop" is two tokens out of a hundred; what matters
clinically is whether the word is in the audio at all, and whether the model invents one where the doctor
stayed silent. So this counts, per arm and per word type:

  ref      occurrences in the reference text
  said     occurrences heard in the ASR hypothesis (capped at ref, so padding a clip with extra words is
           never rewarded)
  missed   ref - said        the notation asked for the word and the audio does not have it
  invented hyp - ref when positive: the model pronounced a punctuation word the text never contained

"invented" is the failure mode that makes the mapping ambiguous from the listener's side, and it is what the
label normalization was for: `. Full stop.` in the text, and nothing else, should produce the sound.

Reads the asr_results.json written by local/asr_score.py (hypotheses keyed by wav path) plus the split's text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WORDS = ('full stop', 'period', 'comma')

# On this domain the ASR hears the phrase but spells it wrong: "Full stop" came back as footstop, free stop,
# fuel stop. Strict matching therefore under-counts what was pronounced. The fuzzy column is an upper bound
# -- it also catches an ordinary English "stop" -- so read the pair: the truth is between them, and the only
# comparison that is safe to make is between arms, which share one ASR and one detector.
FUZZY = {'full stop': r'\b(?:\w+\s)?stop\b', 'period': r'\bperiod\b', 'comma': r'\bcomma\b'}


def count(text: str, fuzzy: bool = False) -> dict:
    t = text.lower()
    return {w: len(re.findall(FUZZY[w] if fuzzy else rf'\b{w}\b', t)) for w in WORDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--asr', type=Path, required=True, help='asr_results.json from local/asr_score.py')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    ref = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
           for l in (args.data_root / args.split / 'text').read_text().splitlines()}
    store = json.loads(args.asr.read_text())
    # hypotheses are keyed by wav path; the utt id ends with the file stem in both sources
    wav_scp = args.data_root / args.split / 'wav.scp'
    stem2utt = ({Path(l.split(maxsplit=1)[1]).stem: l.split(maxsplit=1)[0]
                 for l in wav_scp.read_text().splitlines()} if wav_scp.is_file()
                else {u: u for u in ref})  # a generated corpus names its wavs <utt>.wav, so stem IS the utt

    print(f'{"arm":9s} {"utt":>4} ' + ' '.join(f'{w[:6]:>22s}' for w in WORDS) + '   total  strict  fuzzy')
    for arm, hyp_by_path in store['hyp'].items():
        agg = {w: [0, 0, 0, 0] for w in WORDS}  # per word: ref, said_strict, said_fuzzy, invented_fuzzy
        n = 0
        for path, hyp_text in hyp_by_path.items():
            utt = stem2utt.get(Path(path).stem) or next((u for u in ref if u.endswith(Path(path).stem)), None)
            if utt is None or utt not in ref:
                continue
            n += 1
            r, hs, hf = count(ref[utt]), count(hyp_text), count(hyp_text, fuzzy=True)
            for w in WORDS:
                agg[w][0] += r[w]
                agg[w][1] += min(r[w], hs[w])
                agg[w][2] += min(r[w], hf[w])
                agg[w][3] += max(0, hf[w] - r[w])
        tot = [sum(a[i] for a in agg.values()) for i in range(4)]
        cells = ' '.join(f'{a[1]:3d}/{a[2]:3d} of {a[0]:<3d}(+{a[3]})' for a in agg.values())
        strict = tot[1] / tot[0] * 100 if tot[0] else float('nan')
        fuzzy = tot[2] / tot[0] * 100 if tot[0] else float('nan')
        print(f'{arm:9s} {n:4d} {cells}   {tot[0]:5d} {strict:5.1f}% {fuzzy:5.1f}%')
    print("\nper word: said_strict/said_fuzzy of ref_count (+invented, fuzzy upper bound: the hyp had more "
          'such words than the text asked for). strict = exact spelling, fuzzy = also accepts the ASR mishearing '
          'of the phrase; the invented column only means something compared across arms.')


if __name__ == '__main__':
    main()
