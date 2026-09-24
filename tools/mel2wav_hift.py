#!/usr/bin/env python3
# Standalone CosyVoice3 vocoder: synthesize 24kHz waveform from 80-dim log-mel using CausalHiFTGenerator only.
#
# Input:  .npy/.pt file holding mel of shape (80, T) or (1, 80, T), 50 frames per second (hop 480 @ 24kHz)
#         or a .wav file, which is first converted to mel with the exact training-time mel definition (round-trip test).
# Example:
#   python tools/mel2wav_hift.py mel.npy --output out.wav \
#       --model_dir pretrained_models/Fun-CosyVoice3-0.5B
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torchaudio

# match tools/synthesize_dialogue.py: cosyvoice and vendored Matcha-TTS are not pip-installed
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.append(os.path.join(repo_root, 'third_party', 'Matcha-TTS'))

from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.utils.file_utils import load_wav


def build_hift(model_dir, device):
    with open('{}/cosyvoice3.yaml'.format(model_dir), 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None, 'hifigan': None})
    hift = configs['hift']
    # hift.pt is saved from the HiFiGan wrapper, generator weights carry a 'generator.' prefix
    state_dict = torch.load('{}/hift.pt'.format(model_dir), map_location='cpu', weights_only=True)
    state_dict = {k.replace('generator.', ''): v for k, v in state_dict.items()}
    hift.load_state_dict(state_dict, strict=True)
    hift.to(device).eval()
    return hift, configs['mel_spec_transform1']


def read_mel(path, mel_fn, device):
    if path.endswith(('.npy', '.pt')):
        if path.endswith('.npy'):
            mel = torch.from_numpy(np.load(path))
        else:
            mel = torch.load(path, map_location='cpu', weights_only=True)
        mel = mel.float()
        assert mel.dim() in (2, 3), 'expect mel shape (80, T) or (1, 80, T), got {}'.format(tuple(mel.shape))
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.shape[-1] == 80 and mel.shape[1] != 80:  # tolerate (B, T, 80)
            mel = mel.transpose(1, 2)
        assert mel.shape[1] == 80, 'expect mel shape (80, T) or (1, 80, T), got {}'.format(tuple(mel.shape))
    else:
        mel = mel_fn(load_wav(path, 24000))
        assert mel.shape[1] == 80
    return mel.to(device)


def main():
    parser = argparse.ArgumentParser(description='CosyVoice3 standalone vocoder (mel -> 24kHz wav)')
    parser.add_argument('input', help='mel .npy/.pt ((80, T) or (1, 80, T), 50Hz) or a .wav to round-trip')
    parser.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='dir containing cosyvoice3.yaml and hift.pt')
    parser.add_argument('--output', default=None, help='output wav path, default <input>_hift.wav')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not os.path.exists(args.model_dir):
        args.model_dir = snapshot_download(args.model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hift, mel_fn = build_hift(args.model_dir, device)
    mel = read_mel(args.input, mel_fn, device)

    start_time = time.time()
    with torch.inference_mode():
        speech, _ = hift.inference(mel)  # f0 is predicted from mel inside, no external f0 needed
    speech_len = speech.shape[1] / hift.sampling_rate
    logging.info('mel {}, speech {:.2f}s, rtf {:.3f}'.format(
        tuple(mel.shape), speech_len, (time.time() - start_time) / speech_len))

    output = args.output or os.path.splitext(args.input)[0] + '_hift.wav'
    torchaudio.save(output, speech.cpu(), hift.sampling_rate)
    logging.info('saved {}'.format(output))


if __name__ == '__main__':
    main()
