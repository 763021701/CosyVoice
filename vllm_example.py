import os
import sys

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, 'third_party', 'Matcha-TTS'))

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
import torchaudio

from example import MODES, build_parser, load_audios, resolve_model, synthesize

# Plain / mild / strong Cantonese-accent English. Brackets are CMU phones.
PHONEME_VARIANTS = (
    ('plain', (
        'Specimen labelled left ovarian cyst.  '
        'Submitted were strips of soft tan-coloured tissue measuring on aggregate'
    )),
    ('weak_accent', (
        'Specimen labelled [L][EH1][F] [OW0][W][EH1][R][IY0][AH0][N] [S][IH1][S].  '
        'Submitted were strips of [S][AO1][F] tan-coloured tissue measuring on [AE1][G][R][AH0][G][AH0].'
    )),
    ('strong_accent', (
        'Specimen labelled [L][EH1][F] [OW0][W][EH1][W][IY0][AH0][N] [S][IH1][S].  '
        'Submitted were [S][T][IH1][P][S] of [S][AO1][F] [T][EH1][N]-coloured [T][IH1][S][UW0] measuring on [EH1][G][W][AH0][G][AH0].'
    )),
)
PHONEME_PROMPT_TEXT = (
    'You are a helpful assistant.<|endofprompt|>'
    'Specimen labelled uterus and bilateral ovaries and tubes.  '
    'Submitted was a hysterectomy and bilateral salpingo-oophorectomy specimen.'
)


def synthesize_phoneme(cosyvoice, audio):
    for variant_name, tts_text in PHONEME_VARIANTS:
        set_all_random_seed(0)
        print('=== zero_shot | {} ==='.format(variant_name))
        for i, speech in enumerate(cosyvoice.inference_zero_shot(
                tts_text, PHONEME_PROMPT_TEXT, audio, stream=False, text_frontend=False)):
            out_path = 'vllm_phoneme_{}_{}.wav'.format(variant_name, i)
            torchaudio.save(out_path, speech['tts_speech'], cosyvoice.sample_rate)
            print('saved {}'.format(out_path))


def main():
    parser = build_parser(modes=MODES + ('phoneme',))
    parser.description = 'CosyVoice 推理（vLLM + TensorRT）。参数与 example.py 相同，另外支持 --mode phoneme。'
    args = parser.parse_args()
    audio, prompt_audio = load_audios(args.mode, args.audio)
    cosyvoice = AutoModel(model_dir=resolve_model(args.model), load_trt=True, load_vllm=True, fp16=False)
    if args.mode == 'phoneme':
        synthesize_phoneme(cosyvoice, audio)
        return
    synthesize(cosyvoice, args.mode, audio, prompt_audio, args.text, args.prompt_text, args.instruct_text)


if __name__ == '__main__':
    main()
