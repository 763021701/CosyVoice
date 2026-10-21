#!/usr/bin/env python3
"""Turn a training checkpoint into a directory `CosyVoice3(model_dir)` loads with no extra code.

Why a copy of the weights is unavoidable: `CosyVoiceModel.load` does
`llm.load_state_dict(torch.load('<dir>/llm.pt'), strict=True)` (cosyvoice/cli/model.py:66), while
`epoch_N_whole.pt` carries two bookkeeping keys on top of the 293 parameters (`epoch`, `step`), which
`strict=True` rejects as unexpected keys. So the checkpoint is rewritten without them -- 2.0G per epoch.

Everything else in the model dir is symlinked from the base model, not copied: only the LLM was
fine-tuned, so flow.pt / hift.pt / campplus / speech_tokenizer / cosyvoice3.yaml / asset are byte-identical
to what training started from. Same layout as R1's `data/ft_model/`.

    python local/package_ft_model.py \
        --ckpt exp/cosyvoice3_mixclean/llm/torch_ddp/epoch_1_whole.pt \
        --out  data_mix_clean/ft_ep1

Note the two calls a *user* of the resulting directory still has to make in this environment, before
constructing CosyVoice3, or every clip degenerates (see cosyvoice_train.fix_incremental_decoding):

    from cosyvoice_train import force_qwen2_fp32, fix_incremental_decoding
    force_qwen2_fp32(); fix_incremental_decoding()
"""
import argparse
from pathlib import Path

import torch

REQUIRED = ('cosyvoice3.yaml', 'flow.pt', 'hift.pt', 'campplus.onnx',
            'speech_tokenizer_v3.onnx', 'CosyVoice-BlankEN', 'asset')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True, help='epoch_N_whole.pt')
    ap.add_argument('--base', type=Path,
                    default=Path('/root/autodl-tmp/.cache/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512'),
                    help='base model dir: source of everything except llm.pt')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    extra = sorted(k for k in state if k in ('epoch', 'step'))
    state = {k: v for k, v in state.items() if k not in ('epoch', 'step')}
    ref = torch.load(args.base / 'llm.pt', map_location='cpu', weights_only=True)
    assert set(state) == set(ref), (f'{args.ckpt} differs from base llm.pt: '
                                    f'only ckpt {sorted(set(state) - set(ref))[:3]}, '
                                    f'only base {sorted(set(ref) - set(state))[:3]}')
    assert all(v.dtype == ref[k].dtype for k, v in state.items()), 'dtype mismatch vs base'
    print(f'{args.ckpt}: {len(state)} params, dropped {extra}')

    args.out.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        src = args.base / name
        assert src.exists(), f'base model dir is missing {name}'
        link = args.out / name
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(src.resolve())
    dest = args.out / 'llm.pt'
    torch.save(state, dest)
    print(f'wrote {dest} ({dest.stat().st_size / 1e9:.2f} G) + {len(REQUIRED)} symlinks -> {args.out}')


if __name__ == '__main__':
    main()
