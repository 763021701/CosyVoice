import argparse
import os
import sys

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT_DIR, 'third_party', 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

MODEL_ALIASES = {
    'cosyvoice3': 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    'cosyvoice2': 'iic/CosyVoice2-0.5B',
    'cosyvoice': 'iic/CosyVoice-300M',
}

MODES = ('zero_shot', 'cross_lingual', 'instruct', 'vc')

DEFAULT_TEXTS = {
    'zero_shot': '八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。',
    'cross_lingual': 'You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]',
    'instruct': '好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。',
}
DEFAULT_PROMPT_TEXT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
DEFAULT_INSTRUCT_TEXT = 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>'


def resolve_model(name):
    return MODEL_ALIASES.get(name, name)


def resolve_path(path):
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(_ROOT_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return path


def ensure_endofprompt(text):
    if '<|endofprompt|>' in text:
        return text
    return 'You are a helpful assistant.<|endofprompt|>' + text


def build_parser(modes=MODES):
    parser = argparse.ArgumentParser(
        description='CosyVoice 推理。vc 需要两段音频：源语音在前，目标说话人在后。',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mode', required=True, choices=modes, help='推理模式')
    parser.add_argument('--audio', required=True, nargs='+', help='提示音频。vc 时传两段：源语音 目标说话人')
    parser.add_argument('--model', default='cosyvoice3', help='cosyvoice3 / cosyvoice2 / cosyvoice，或模型目录 / ModelScope id')
    parser.add_argument('--text', default=None, help='要合成的文本；vc 忽略')
    parser.add_argument('--prompt-text', default=DEFAULT_PROMPT_TEXT, help='提示音频对应文本，仅 zero_shot 使用')
    parser.add_argument('--instruct-text', default=DEFAULT_INSTRUCT_TEXT, help='控制指令，仅 instruct 使用')
    return parser


def load_audios(mode, audio_args):
    paths = [resolve_path(path) for path in audio_args]
    if mode == 'vc':
        if len(paths) != 2:
            raise SystemExit('vc requires two audios: --audio SOURCE TARGET_SPEAKER')
        source, prompt = paths
    else:
        if len(paths) != 1:
            raise SystemExit('{} requires one audio: --audio PROMPT'.format(mode))
        source, prompt = paths[0], None
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError('audio not found: {}'.format(path))
    return source, prompt


def synthesize(cosyvoice, mode, audio, prompt_audio, text, prompt_text, instruct_text):
    model_name = cosyvoice.__class__.__name__
    if text is None:
        text = DEFAULT_TEXTS.get(mode, DEFAULT_TEXTS['zero_shot'])
    if model_name == 'CosyVoice3':
        if mode == 'zero_shot':
            prompt_text = ensure_endofprompt(prompt_text)
        elif mode == 'cross_lingual':
            text = ensure_endofprompt(text)
        elif mode == 'instruct':
            instruct_text = ensure_endofprompt(instruct_text)

    if mode == 'zero_shot':
        outputs = cosyvoice.inference_zero_shot(text, prompt_text, audio, stream=False)
    elif mode == 'cross_lingual':
        outputs = cosyvoice.inference_cross_lingual(text, audio, stream=False)
    elif mode == 'instruct':
        if model_name == 'CosyVoice':
            outputs = cosyvoice.inference_instruct(text, '中文女', instruct_text, stream=False)
        else:
            outputs = cosyvoice.inference_instruct2(text, instruct_text, audio, stream=False)
    elif mode == 'vc':
        outputs = cosyvoice.inference_vc(audio, prompt_audio, stream=False)
    else:
        raise ValueError('unknown mode: {}'.format(mode))

    for i, speech in enumerate(outputs):
        out_path = '{}_{}.wav'.format(mode, i)
        torchaudio.save(out_path, speech['tts_speech'], cosyvoice.sample_rate)
        print('saved {}'.format(out_path))


def main():
    args = build_parser().parse_args()
    audio, prompt_audio = load_audios(args.mode, args.audio)
    cosyvoice = AutoModel(model_dir=resolve_model(args.model))
    synthesize(cosyvoice, args.mode, audio, prompt_audio, args.text, args.prompt_text, args.instruct_text)


if __name__ == '__main__':
    main()
