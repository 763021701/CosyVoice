#!/usr/bin/env python3
"""Second reader for the punctuation words: faster-whisper fine-tuned on Cantonese-Yue-English.

Qwen3-ASR is the primary scorer, but on this domain it mis-transcribes the real dictation badly ("Foveolar
hyperplasia is noted" -> "除非呢个place是不footstop"), so a low punct-word detection rate on an arm could mean
either that the word was not pronounced or that the reader cannot hear it through this accent. A decoder that
was trained on Cantonese-English code-switching is the control for that: if the arm ordering holds under both
readers, the drop is in the audio, not in the transcription.

Emits the same {'hyp': {arm: {wav_path: text}}} shape as local/asr_score.py so local/punct_metric.py reads it
unchanged. Run in the whisper_ft env (it is the one with faster_whisper + ctranslate2):

    /root/autodl-tmp/miniconda3/envs/whisper_ft/bin/python -I local/fw_hyp.py ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from faster_whisper import WhisperModel

MODEL = ('/root/autodl-tmp/.cache/huggingface/hub/'
         'models--JackyHoCL--whisper-large-v3-turbo-cantonese-yue-english-ct2/snapshots')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--sample', type=float, default=0.0,
                    help='fraction of the intersection to read, same md5 rule as asr_score.py -- use a smaller '
                         'sample here: this reader only has to confirm the ordering between arms, not the rate')
    ap.add_argument('--model-dir', default=MODEL)
    args = ap.parse_args()

    snap = Path(args.model_dir)
    if not (snap / 'model.bin').is_file():  # resolve .../snapshots/<sha>/model.bin
        snap = sorted(p for p in snap.iterdir() if (p / 'model.bin').is_file())[-1]
    print(f'reader: {snap.name}')
    model = WhisperModel(str(snap), device='cuda', compute_type='float16')

    ref = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
           for l in (args.data_root / args.split / 'text').read_text().splitlines()}
    wav_scp = args.data_root / args.split / 'wav.scp'
    real = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
            for l in wav_scp.read_text().splitlines()} if wav_scp.is_file() else {}
    keep = [u for u in sorted(ref) if all((args.out_root / a / f'{u}.wav').is_file() for a in args.arms)]
    if args.sample:
        keep = [u for u in keep if int(hashlib.md5(u.encode()).hexdigest()[:8], 16) % 1000 < int(args.sample * 1000)]
    print(f'{len(keep)} utt in the intersection of arms {args.arms}'
          + (f' (--sample {args.sample:.2f})' if args.sample else ''))

    def read(path):
        segs, _ = model.transcribe(str(path), beam_size=5, task='transcribe', condition_on_previous_text=False)
        return ' '.join(s.text.strip() for s in segs).strip()

    hyp = {}
    readers = ([('real', {u: real[u] for u in keep})] if real and all(u in real for u in keep) else []) + \
              [(a, {u: str(args.out_root / a / f'{u}.wav') for u in keep}) for a in args.arms]
    for arm, paths in readers:
        hyp[arm] = {p: read(p) for p in paths.values()}
        print(f'{arm}: {len(hyp[arm])} read')
    (args.out_root / 'asr_fw.json').write_text(json.dumps({'hyp': hyp}, indent=1, ensure_ascii=False) + '\n')
    print(f'-> {args.out_root / "asr_fw.json"}')


if __name__ == '__main__':
    main()
