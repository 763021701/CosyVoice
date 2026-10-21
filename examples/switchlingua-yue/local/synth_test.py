#!/usr/bin/env python3
"""Tier-2 synthesis: render every text of a split once per llm checkpoint, with a same-item voice prompt.

The checkpoint is swapped on the already-loaded model (load_state_dict into cosyvoice.model.llm) rather than
rebuilding CosyVoice3 per checkpoint: flow/hift are identical for every arm, so loading them once keeps the
three arms under exactly one vocoder and one flow, which is what makes the comparison about the LLM.

prompt_text carries the instruct prefix inside it, the way example.py does for CosyVoice3. This is not
cosmetic: Qwen2LM/CosyVoice3LM.inference build [sos, prompt_text ++ text, task_id, prompt_speech_token] and
have no separate slot for the instruct, unlike the training path which reads batch['instruct_token']. Without
the prefix, inference would not match the format the model was fine-tuned on.

text_frontend=False on purpose: the default normalizer rewrites numbers/symbols and would change the
code-switched surface form we are trying to measure. Our texts are single already-punctuated sentences.

Writes out_root/<ckpt>/<utt>.wav plus a manifest with the real and synthesized durations, which is the
duration diagnostic (does the fine-tuned LLM stretch or compress speech relative to real).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "third_party" / "Matcha-TTS")]

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

# same patch the training launcher applies: transformers 5 hands back Qwen2 in bf16 while the rest of
# CosyVoice3LM is fp32, which kills llm_job mid-generation and leaves tts() to crash later in hift with a
# 3-frame mel. Shared rather than duplicated so the trained and the evaluated model see one dtype.
from cosyvoice_train import fix_incremental_decoding, force_qwen2_fp32
from cosyvoice.cli.cosyvoice import CosyVoice3

INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def seconds(path: str) -> float:
    # soundfile rather than torchaudio.info: the latter is gone in this torchaudio
    info = sf.info(path)
    return info.frames / info.samplerate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True, help='base CosyVoice3 dir: provides flow/hift/onnx/yaml')
    ap.add_argument('--llm', action='append', required=True, metavar='NAME=PATH')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='only synthesize the first N (for a timing smoke)')
    ap.add_argument('--shard', default='', metavar='K/N',
                    help='take only every Nth utterance (offset K) of the sorted list, so several workers can '
                          'split one cell without ever generating the same utt. NOTE: <arm>.manifest.json and '
                          '<arm>.failed.json then describe only this shard -- rerun the cell with no --shard '
                          'once all shards are done to rewrite them over the whole corpus (every clip is '
                          'already on disk, so that pass only reads headers)')
    ap.add_argument('--seed', type=int, default=1986,
                    help='ras_sampling draws from a multinomial; reseed per arm so a run is reproducible')
    args = ap.parse_args()

    pairs_file = args.data_root / f'{args.split}_prompt_pairs.json'
    pairs = json.loads(pairs_file.read_text())
    text = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
            for l in (args.data_root / args.split / 'text').read_text().splitlines()}
    # a text-only generation corpus has no reference audio at all: empty map => real_sec stays None downstream
    wav_scp = args.data_root / args.split / 'wav.scp'
    wav = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
           for l in wav_scp.read_text().splitlines()} if wav_scp.is_file() else {}
    order = sorted(pairs)
    if args.shard:
        k, n = (int(x) for x in args.shard.split('/'))
        order = order[k::n]
    if args.limit:
        order = order[:args.limit]

    # both must precede construction: fp32 so llm.pt loads unrounded, and the decoding patch because without
    # it every generated clip collapses into a repetition loop (see cosyvoice_train.fix_incremental_decoding)
    force_qwen2_fp32()
    fix_incremental_decoding()
    cosyvoice = CosyVoice3(args.model_dir, load_trt=False, load_vllm=False, fp16=False)
    print(f'{len(pairs)} utt in split, {len(order)} to synthesize, sample_rate {cosyvoice.sample_rate}')

    for spec in args.llm:
        name, path = spec.split('=', 1)
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = {k: v for k, v in state.items() if k not in ('epoch', 'step')}
        cosyvoice.model.llm.load_state_dict(state, strict=True)
        cosyvoice.model.llm.eval()
        torch.manual_seed(args.seed)  # same starting stream for every arm; only the LLM differs

        out = args.out_root / name
        out.mkdir(parents=True, exist_ok=True)
        manifest, failed = [], []
        for utt in tqdm(order, desc=name):
            p = pairs[utt]
            dest = out / f'{utt}.wav'
            if dest.is_file():  # resume: a crashed run continues rather than redoing everything
                manifest.append({'utt': utt, 'synth_wav': str(dest),
                                 'synth_sec': round(seconds(str(dest)), 3),
                                 'real_sec': round(seconds(wav[utt]), 3) if utt in wav else None,
                                 'fallback': p['fallback'],
                                 'same_item': p['same_item'], 'text': text[utt]})
                continue
            try:
                chunks = [o['tts_speech'] for o in cosyvoice.inference_zero_shot(
                    text[utt], INSTRUCT + p['prompt_text'], p['prompt_wav'],
                    stream=False, text_frontend=False)]
                speech = torch.concat(chunks, dim=1)
            except Exception as e:  # noqa: BLE001 - a 30k-clip run must survive anything one clip can raise
                # One degenerate utterance must not kill a 2h run. ep9 raised "sampling reaches max_trials
                # 100 and still get eos when ignore_eos is True" inside llm_job; the thread died, tts()
                # carried on with zero tokens and blew up later in hift on a 3-frame mel.
                failed.append({'utt': utt, 'error': str(e).strip().splitlines()[-1][:200]})
                continue
            torchaudio.save(str(dest), speech, cosyvoice.sample_rate)
            manifest.append({'utt': utt, 'synth_wav': str(dest),
                             'synth_sec': round(speech.shape[1] / cosyvoice.sample_rate, 3),
                             'real_sec': round(seconds(wav[utt]), 3) if utt in wav else None,
                             'fallback': p['fallback'], 'same_item': p['same_item'],
                             'text': text[utt]})
        if failed:
            (args.out_root / f'{name}.failed.json').write_text(
                json.dumps(failed, indent=1, ensure_ascii=False) + '\n')
            print(f'  {name}: {len(failed)} utt failed, see {name}.failed.json')
        elif not args.shard and (args.out_root / f'{name}.failed.json').is_file():
            # a clean rerun leaves the previous run's list behind, and both `status` and the QC completeness
            # gate read it as if it were current -- 15 utts reported failed forever after they had succeeded.
            # Only a whole-cell pass may remove it: under --shard each worker sees part of the corpus.
            (args.out_root / f'{name}.failed.json').unlink()
            print(f'  {name}: all utts regenerated clean, removed the stale {name}.failed.json')
        (args.out_root / f'{name}.manifest.json').write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False) + '\n')
        syn = sum(m['synth_sec'] for m in manifest)
        # a generation corpus has no real reference: report synth hours alone rather than a fake ratio
        real = sum(m['real_sec'] for m in manifest if m['real_sec'] is not None)
        tail = (f' vs {real:.0f}s real -> duration ratio {syn / real:.3f}' if real else
                f' ({syn / 3600:.2f} h synth, no reference audio in this root)')
        print(f'  {name}: {len(manifest)} wav, {syn:.0f}s synth{tail}')


if __name__ == '__main__':
    main()
