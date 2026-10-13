#!/usr/bin/env python3
"""Runtime patches for this machine's torch 2.11 / transformers 5, plus the trainer launcher.

The first two are what the trainer needs; `fix_incremental_decoding` is only needed to *generate* and is
imported by synth_test.py -- training runs the whole sequence in one forward, so it never hits it.
All three are environment drift, not CosyVoice bugs, so they live here rather than in the shared cosyvoice/
package (which every other user of this checkout reads):

 1. cosyvoice_join() reads the barrier timeout from `group_join.options._timeout`; torch 2.11's
    ProcessGroup no longer has `.options`. The replacement takes it from info_dict["timeout"], which is
    the same value train.py already passes into dist.new_group() for that group.
 2. transformers 5 loads CosyVoice-BlankEN in the config dtype (bfloat16); the recipe was written against
    transformers 4, which loaded fp32. Qwen2Encoder is cast back to fp32 after loading so training keeps
    fp32 weights under AMP -- GradScaler cannot unscale bf16 grads -- and llm.pt loads unrounded.

Patch 1 must be set on cosyvoice.utils.executor, which imports the name directly (executor.py:23);
patching train_utils would leave executor's own binding untouched.

Invoke exactly as you would train.py:
  torchrun --nproc_per_node=1 local/cosyvoice_train.py --model llm --config ... etc.
"""

from __future__ import annotations

import datetime
import logging
import os
import runpy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "third_party" / "Matcha-TTS")]

import torch.distributed as dist


def cosyvoice_join(group_join, info_dict):
    if info_dict["batch_idx"] != 0:
        try:
            dist.monitored_barrier(group=group_join,
                                   timeout=datetime.timedelta(seconds=info_dict["timeout"]))
            return False
        except RuntimeError as e:
            logging.info(f"Detected uneven workload distribution: {e}\n"
                         "Break current worker to manually join all workers")
            return True
    return False


def force_qwen2_fp32():
    import cosyvoice.llm.llm as llm_module

    original = llm_module.Qwen2Encoder.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.float()

    llm_module.Qwen2Encoder.__init__ = init


def fix_incremental_decoding():
    """Repair step-by-step decoding, which transformers 5 silently breaks.

    Qwen2Encoder.forward_one_step hands the model `masks[:, -1, :]` and no cache_position. Once the cache is
    non-empty lm_input is a single token, so that mask is 1x1 while the cache holds the whole prefix. v4
    inferred the position from the past length; v5 does not, so every step is scored at the same position and
    generation collapses into a repetition loop. Measured on CosyVoice-BlankEN: the incremental last hidden
    state differed from the full-sequence one by max abs 45.07 before this patch, 7.0e-04 after.

    It is invisible in teacher-forced loss (forward() runs the whole sequence in one pass), which is why the
    fine-tune looked healthy while its audio was unusable. For one unpadded sequence all-ones is exactly what
    `masks[:, -1, :]` was meant to be, so nothing else changes.
    """
    import cosyvoice.llm.llm as llm_module

    def forward_one_step(self, xs, masks, cache=None):
        past = cache.get_seq_length() if cache is not None else 0
        new = {'attention_mask': torch.ones(1, past + xs.size(1), dtype=torch.bool, device=xs.device)}
        if cache is not None:
            new['past_key_values'] = cache
            new['cache_position'] = torch.arange(past, past + xs.size(1), device=xs.device)
        outs = self.model(inputs_embeds=xs, output_hidden_states=True, return_dict=True,
                          use_cache=True, **new)
        return outs.hidden_states[-1], outs.past_key_values

    llm_module.Qwen2Encoder.forward_one_step = forward_one_step


if __name__ == "__main__":
    import cosyvoice.utils.executor as executor

    executor.cosyvoice_join = cosyvoice_join
    force_qwen2_fp32()
    train_py = str(REPO_ROOT / "cosyvoice" / "bin" / "train.py")
    sys.argv[0] = train_py
    runpy.run_path(train_py, run_name="__main__")
