import os
import sys

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT_DIR, 'third_party', 'Matcha-TTS'))

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm
import torchaudio

# --- shared config for medical English accent experiments ---
MODEL_DIR = 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512'
PROMPT_WAV = '/root/autodl-tmp/workspace/dataset/Test_Samples/Sample1_0139_0149.wav'

# Plain English
TTS_TEXT = (
    'Specimen labelled left ovarian cyst.  '
    'Submitted were strips of soft tan-coloured tissue measuring on aggregate'
)
# Mild Cantonese-accent English (CMU phones)
TTS_TEXT_WEAK_ACENT = (
    'Specimen labelled [L][EH1][F] [OW0][W][EH1][R][IY0][AH0][N] [S][IH1][S].  '
    'Submitted were strips of [S][AO1][F] tan-coloured tissue measuring on [AE1][G][R][AH0][G][AH0].'
)
# Strong Cantonese-accent English (CMU phones)
TTS_TEXT_STRONG_ACENT = (
    'Specimen labelled [L][EH1][F] [OW0][W][EH1][W][IY0][AH0][N] [S][IH1][S].  '
    'Submitted were [S][T][IH1][P][S] of [S][AO1][F] [T][EH1][N]-coloured [T][IH1][S][UW0] measuring on [EH1][G][W][AH0][G][AH0].'
)

ZERO_SHOT_PROMPT_TEXT = (
    'You are a helpful assistant.<|endofprompt|>'
    'Specimen labelled uterus and bilateral ovaries and tubes.  '
    'Submitted was a hysterectomy and bilateral salpingo-oophorectomy specimen.'
)


def cosyvoice2_example():
    """ CosyVoice2 vllm usage
    """
    cosyvoice = AutoModel(model_dir='iic/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot(
                '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
                '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False)):
            continue


def cosyvoice3_example():
    """ CosyVoice3 vllm usage
    """
    cosyvoice = AutoModel(model_dir='FunAudioLLM/Fun-CosyVoice3-0.5B-2512', load_trt=True, load_vllm=True, fp16=False)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot(
                '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
                'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                './asset/zero_shot_prompt.wav', stream=False)):
            continue


def my_cosyvoice_phoneme_compare():
    """ A/B/C: plain vs weak vs strong CMU spellings (zero_shot only).

    Outputs: vllm_phoneme_{plain,weak_accent,strong_accent}_0.wav
    """
    cosyvoice = AutoModel(model_dir=MODEL_DIR, load_trt=True, load_vllm=True, fp16=False)

    variants = [
        ('plain', TTS_TEXT),
        ('weak_accent', TTS_TEXT_WEAK_ACENT),
        ('strong_accent', TTS_TEXT_STRONG_ACENT),
    ]

    for variant_name, tts_text in variants:
        set_all_random_seed(0)
        print(f'=== zero_shot | {variant_name} ===')
        for i, j in enumerate(cosyvoice.inference_zero_shot(
                tts_text, ZERO_SHOT_PROMPT_TEXT, PROMPT_WAV,
                stream=False, text_frontend=False)):
            out_path = f'vllm_phoneme_{variant_name}_{i}.wav'
            torchaudio.save(out_path, j['tts_speech'], cosyvoice.sample_rate)
            print(f'saved {out_path}')


def main():
    # cosyvoice2_example()
    # cosyvoice3_example()
    my_cosyvoice_phoneme_compare()


if __name__ == '__main__':
    main()
