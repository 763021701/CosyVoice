#!/usr/bin/env python3
"""Tier-1 evaluation: teacher-forced CE loss and top-1 speech-token accuracy, one or more llm.pt
compared on the same batches. Answers "did the fine-tuned LLM get better at emitting the token
sequence the tokenizer actually produces for this audio" -- a held-out measure that needs no
synthesis and no ASR.

It is CosyVoice3LM.forward's own numbers: CE against the real tokens and th_accuracy over the
non-IGNORE positions, i.e. exactly the training objective evaluated off the training split.

Two things make the comparison honest rather than incidental:
  * mix_ratio is widened so prepare_lm_input_target's bistream branch (llm.py:323,
    `random() < 0.5 and speech_len/text_len > mix_ratio[1]/mix_ratio[0]`) can never fire. Without this
    each batch would score under a randomly chosen input format and the numbers would not be comparable.
    Forcing unistream matches what non-streaming inference actually runs.
  * every checkpoint sees the same DataLoader pass (mode='dev' => no shuffle), and the loss is averaged
    weighted by speech_token_len, since forward() returns a length-normalized scalar per batch.

Checkpoints written by cosyvoice/bin/train.py carry 'epoch'/'step' alongside the tensors; they are
dropped here the way cosyvoice/bin/average_model.py drops them.

Do not set the `onnx_path` env var: it would build online extractors we never use (parquet has tokens).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "third_party" / "Matcha-TTS")]

import torch
from hyperpyyaml import load_hyperpyyaml

# must run before load_hyperpyyaml instantiates configs['llm'], so the checkpoints are compared with the
# same fp32 backbone the trainer used
from cosyvoice_train import force_qwen2_fp32
from cosyvoice.dataset.dataset import Dataset


def load_llm(configs, ckpt: str, device):
    model = copy.deepcopy(configs['llm'])
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    state = {k: v for k, v in state.items() if k not in ('epoch', 'step')}
    model.load_state_dict(state, strict=True)
    # widen the bistream gate: speech/text can never exceed 1e6/5, so every sample is scored as unistream
    model.mix_ratio = [5, 10 ** 6]
    return model.to(device).eval()


@torch.inference_mode()
def score(model, batches, device):
    total_loss, total_acc, total_w = 0.0, 0.0, 0
    # same autocast the trainer uses (train_utils.batch_forward:250 --use_amp => bf16); the Qwen2 backbone
    # loads as bf16 while the pipeline emits fp32 features, so a bare forward() would die on the dtype mix
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for batch in batches:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(batch, device)
            w = int(batch['speech_token_len'].sum())
            total_loss += float(out['loss']) * w
            total_acc += float(out['acc']) * w
            total_w += w
    # forward() returns a per-batch length-normalized scalar, so batches are weighted by token count
    return total_loss / total_w, total_acc / total_w, total_w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=Path, required=True)
    ap.add_argument('--qwen-pretrain-path', required=True)
    ap.add_argument('--data', required=True, help='parquet data.list, e.g. data/test/parquet/data.list')
    ap.add_argument('--llm', action='append', required=True, metavar='NAME=PATH',
                    help='checkpoint to score; repeat to compare, first column is the name given here')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    force_qwen2_fp32()
    configs = load_hyperpyyaml(args.conf.open(), overrides={'qwen_pretrain_path': args.qwen_pretrain_path})
    device = torch.device(args.device)

    ds = Dataset(args.data, data_pipeline=configs['data_pipeline'], mode='dev', gan=False, dpo=False,
                 shuffle=False, partition=False)
    batches = list(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0))
    n_utt = sum(len(b['utts']) for b in batches)
    print(f'data {args.data}: {n_utt} utt, {len(batches)} batches')

    rows = []
    for spec in args.llm:
        name, path = spec.split('=', 1)
        model = load_llm(configs, path, device)
        loss, acc, w = score(model, batches, device)
        rows.append((name, loss, acc))
        print(f'  {name:12s} CE {loss:.4f}  top1 {acc * 100:.2f}%  ({w} tokens)')
        del model
        torch.cuda.empty_cache()

    base_name, base_loss, base_acc = rows[0]
    print(f'\nvs {base_name}:')
    for name, loss, acc in rows[1:]:
        print(f'  {name:12s} dCE {loss - base_loss:+.4f}  d_top1 {(acc - base_acc) * 100:+.2f}pp')


if __name__ == '__main__':
    main()
