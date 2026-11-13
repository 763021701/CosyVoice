#!/usr/bin/env python3
"""Does a continuation fine-tune still follow the prompt, or has it collapsed to one reading?

For each held-out target, synthesize the same text twice: once with another utterance of the same
speaker, once with a different speaker. Flow and HiFT stay the base model. Two numbers that move if
the LLM still uses the prompt:

  rate_gap     |chars/sec(synth) - chars/sec(its own prompt)|, same-speaker prompt minus other-speaker
               prompt. Negative means the matched prompt pulled the rate closer than the other one did.
  hist_cos     bag-of-token cosine between the two synths of the same text. Lower means the two prompts
               produced different readings. The real-audio calibration is printed beside it.

Pathology ASR cannot score this: its floor on real audio is already past 86%.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "third_party" / "Matcha-TTS"), str(Path(__file__).resolve().parent)]

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio
import whisper

from cosyvoice_train import fix_incremental_decoding, force_qwen2_fp32
from cosyvoice.cli.cosyvoice import CosyVoice3

INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def read_scp(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            k, v = line.split(maxsplit=1)
            out[k] = v
    return out


def seconds(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def hist_cos(a, b) -> float:
    ca, cb = collections.Counter(a), collections.Counter(b)
    keys = set(ca) | set(cb)
    dot = sum(ca[k] * cb[k] for k in keys)
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    return dot / (na * nb) if na and nb else float("nan")


def tokens_of(path: str, session) -> list:
    audio, sr = torchaudio.load(path, backend="soundfile")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    if audio.shape[1] / 16000 > 30:
        return []
    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    return session.run(None, {
        session.get_inputs()[0].name: feat.detach().cpu().numpy(),
        session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
    })[0].flatten().tolist()


def jobs_for(split_dir: Path, per_speaker: int, pick: str) -> list[dict]:
    text = read_scp(split_dir / "text")
    wav = read_scp(split_dir / "wav.scp")
    spk = read_scp(split_dir / "utt2spk")
    by = collections.defaultdict(list)
    for utt, speaker in spk.items():
        by[speaker].append(utt)
    # SwitchLingua writes utt2spk as the identity. The item prefix is the reader, same as
    # make_continuation_parquet.py and build_prompt_pairs.py. A real speaker map is kept when any
    # speaker has two utterances.
    if max((len(v) for v in by.values()), default=0) < 2:
        by = collections.defaultdict(list)
        for utt in spk:
            by[utt.split("_", 1)[0]].append(utt)
    speakers = sorted(s for s, members in by.items() if len(members) >= 2)
    assert len(speakers) >= 2, "need two speakers to build the other-prompt contrast"
    chosen = []
    for speaker in speakers:
        members = sorted(by[speaker], key=lambda u: len(text[u]))
        if pick == "shortest":
            take = members[:per_speaker]
        else:
            mid = len(members) // 2
            start = max(0, min(mid - per_speaker // 2, len(members) - per_speaker))
            take = members[start:start + per_speaker]
        chosen.extend((speaker, utt) for utt in take)
    rows = []
    for speaker, utt in chosen:
        mates = [u for u in by[speaker] if u != utt]
        other_speaker = speakers[(speakers.index(speaker) + 1) % len(speakers)]
        if pick == "shortest":
            same = min(mates, key=lambda u: len(text[u]))
            other = min(by[other_speaker], key=lambda u: len(text[u]))
        else:
            same = min(mates, key=lambda u: abs(len(text[u]) - len(text[utt])))
            other = min(by[other_speaker], key=lambda u: abs(len(text[u]) - len(text[utt])))
        rows.append({
            "utt": utt, "spk": speaker, "text": text[utt],
            "same_utt": same, "same_wav": wav[same], "same_text": text[same],
            "other_utt": other, "other_wav": wav[other], "other_text": text[other],
            "real_wav": wav[utt],
        })
    return rows


def synthesize(cosyvoice, row, which, dest: Path):
    if dest.is_file():
        return
    prompt_text = row["same_text"] if which == "same" else row["other_text"]
    prompt_wav = row["same_wav"] if which == "same" else row["other_wav"]
    chunks = [o["tts_speech"] for o in cosyvoice.inference_zero_shot(
        row["text"], INSTRUCT + prompt_text, prompt_wav, stream=False, text_frontend=False)]
    torchaudio.save(str(dest), torch.concat(chunks, dim=1), cosyvoice.sample_rate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--llm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-speaker", type=int, default=4)
    ap.add_argument("--pick", choices=("shortest", "median"), default="shortest",
                    help="which utterances of each speaker to hold out. median avoids the very short clips")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1986)
    args = ap.parse_args()

    rows = jobs_for(args.data_root / args.split, args.per_speaker, args.pick)
    print(f"{len(rows)} targets")
    force_qwen2_fp32()
    fix_incremental_decoding()
    cosyvoice = CosyVoice3(args.model_dir, load_trt=False, load_vllm=False, fp16=False)
    for spec in args.llm:
        name, path = spec.split("=", 1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if k not in ("epoch", "step")}
        cosyvoice.model.llm.load_state_dict(state, strict=True)
        cosyvoice.model.llm.eval()
        torch.manual_seed(args.seed)
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        for row in rows:
            for which in ("same", "other"):
                synthesize(cosyvoice, row, which, out / f"{row['utt']}__{which}.wav")
        print(f"synthesized {name}")

    option = onnxruntime.SessionOptions()
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(Path(args.model_dir) / "speech_tokenizer_v3.onnx"),
        sess_options=option, providers=["CPUExecutionProvider"])
    real = torch.load(args.data_root / args.split / "utt2speech_token.pt", map_location="cpu", weights_only=True)

    summary = {}
    print(f'{"arm":12} {"n":>3} {"rate_gap":>9} {"cos_same_other":>14} {"cos_same_real":>13} {"cos_other_real":>14}')
    for spec in args.llm:
        name = spec.split("=", 1)[0]
        rate_gap, cos_so, cos_sr, cos_or = [], [], [], []
        detail = []
        for row in rows:
            same_p = args.out / name / f"{row['utt']}__same.wav"
            other_p = args.out / name / f"{row['utt']}__other.wav"
            same_tok = tokens_of(str(same_p), session)
            other_tok = tokens_of(str(other_p), session)
            if not same_tok or not other_tok:
                continue
            nchar = max(len(row["text"]), 1)
            cps_same = nchar / seconds(str(same_p))
            cps_other = nchar / seconds(str(other_p))
            cps_ps = len(row["same_text"]) / seconds(row["same_wav"])
            cps_po = len(row["other_text"]) / seconds(row["other_wav"])
            rate_gap.append(abs(cps_same - cps_ps) - abs(cps_other - cps_po))
            cos_so.append(hist_cos(same_tok, other_tok))
            real_tok = real[row["utt"]]
            real_tok = real_tok.tolist() if hasattr(real_tok, "tolist") else list(real_tok)
            cos_sr.append(hist_cos(same_tok, real_tok))
            cos_or.append(hist_cos(other_tok, real_tok))
            detail.append({
                "utt": row["utt"], "spk": row["spk"],
                "rate_gap": rate_gap[-1], "cos_same_other": cos_so[-1],
                "cos_same_real": cos_sr[-1], "cos_other_real": cos_or[-1],
            })
        def avg(xs):
            return sum(xs) / len(xs) if xs else float("nan")
        summary[name] = {
            "n": len(rate_gap),
            "rate_gap": avg(rate_gap),
            "cos_same_other": avg(cos_so),
            "cos_same_real": avg(cos_sr),
            "cos_other_real": avg(cos_or),
            "rate_follow": sum(1 for x in rate_gap if x < 0),
            "detail": detail,
        }
        s = summary[name]
        print(f'{name:12} {s["n"]:3d} {s["rate_gap"]:9.3f} {s["cos_same_other"]:14.3f} '
              f'{s["cos_same_real"]:13.3f} {s["cos_other_real"]:14.3f}')
    (args.out / "steer.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, ensure_ascii=False) + "\n")
    print(f"-> {args.out / 'steer.json'}")


if __name__ == "__main__":
    main()
