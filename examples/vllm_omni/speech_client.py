#!/usr/bin/env python3
"""CosyVoice3 的 vLLM-Omni 服务客户端。

配套启动脚本默认是离线合成：NO_ASYNC_CHUNK=1，max_num_seqs=8。
一次 POST /v1/audio/speech/batch，body 里放 8 条 items。服务端按
max_num_seqs 组批，再多塞几条吞吐几乎不涨。

CosyVoice3 没有预置音色，每条请求都是声音克隆，必须带 ref_audio 和 ref_text。
服务端会在 ref_text 里没有 <|endofprompt|> 时自动加上
``You are a helpful assistant.<|endofprompt|>``。

默认就是这 8 条：
    python examples/vllm_omni/speech_client.py

单条：
    python examples/vllm_omni/speech_client.py --mode single --text "收到好友从远方寄来的生日礼物。"
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF_AUDIO = ROOT / "asset" / "zero_shot_prompt.wav"
DEFAULT_REF_TEXT = "希望你以后能够做的比我还好呦。"
# 和服务器 max_num_seqs=8 对齐。这 8 句是吞吐测量用的同一批文本。
DEFAULT_TEXTS = (
    "收到好友从远方寄来的生日礼物，那份意外的惊喜让我心中充满了甜蜜的快乐。",
    "今天天气很好，我们一起去公园散步吧。",
    "语音合成可以把文字变成自然的声音。",
    "并发请求会由服务端按最大序列数自动组批。",
    "标本显示左侧卵巢囊肿，送检组织为浅褐色软组织条索。",
    "患者昨日开始发热，体温最高三十八度五，伴有轻微咳嗽。",
    "请用自然的语速读出下面这段检查结论。",
    "磁共振未见明显占位，建议结合临床继续观察。",
)


def encode_audio(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"reference audio not found: {path}")
    mime = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(path.suffix.lower(), "audio/wav")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def post_json(url: str, body: dict, timeout: float) -> tuple[bytes, str]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}\n{detail}") from exc


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    print(f"saved {path} ({len(payload)} bytes)")


def selected_texts(args) -> list[str]:
    if args.text:
        return list(args.text)
    if args.mode == "batch":
        return list(DEFAULT_TEXTS)
    return [DEFAULT_TEXTS[0]]


def run_single(args, ref_audio: str) -> None:
    text = selected_texts(args)[0]
    body = {
        "model": args.model,
        "input": text,
        "ref_audio": ref_audio,
        "ref_text": args.ref_text,
        "response_format": "pcm" if args.stream else args.response_format,
    }
    if args.stream:
        body["stream"] = True
        body["stream_format"] = "audio"
    url = args.api_base.rstrip("/") + "/v1/audio/speech"
    print(f"POST {url}")
    print(f"text: {text}")
    started = time.perf_counter()
    payload, content_type = post_json(url, body, args.timeout)
    elapsed = time.perf_counter() - started
    suffix = "pcm" if args.stream else args.response_format
    out = Path(args.out_dir) / f"single.{suffix}"
    write_bytes(out, payload)
    print(f"single request finished in {elapsed:.2f}s content_type={content_type}")


def run_batch(args, ref_audio: str) -> None:
    texts = selected_texts(args)
    body = {
        "model": args.model,
        "ref_audio": ref_audio,
        "ref_text": args.ref_text,
        "response_format": args.response_format,
        "items": [{"input": text} for text in texts],
    }
    url = args.api_base.rstrip("/") + "/v1/audio/speech/batch"
    print(f"POST {url}")
    print(f"batch size: {len(texts)}")
    started = time.perf_counter()
    payload, _ = post_json(url, body, args.timeout)
    elapsed = time.perf_counter() - started
    result = json.loads(payload.decode("utf-8"))
    print(
        f"batch {result.get('id')} finished in {elapsed:.2f}s "
        f"succeeded={result.get('succeeded')} failed={result.get('failed')}"
    )
    out_dir = Path(args.out_dir)
    for item in result.get("results", []):
        index = item.get("index")
        if item.get("status") != "success":
            print(f"item {index} failed: {item.get('error')}")
            continue
        audio = base64.b64decode(item["audio_data"])
        write_bytes(out_dir / f"batch_{index}.{args.response_format}", audio)
        usage = item.get("usage") or {}
        if usage:
            print(
                f"item {index} tokens input={usage.get('input_tokens')} "
                f"output={usage.get('output_tokens')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 vLLM-Omni speech client")
    parser.add_argument("--api-base", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="cosyvoice3", help="must match --served-model-name")
    parser.add_argument("--mode", choices=("single", "batch"), default="batch")
    parser.add_argument("--text", action="append", default=None, help="text to synthesize; repeat for batch mode")
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--response-format", default="wav", choices=("wav", "pcm", "flac", "mp3", "opus"))
    parser.add_argument("--stream", action="store_true", help="single mode only: raw PCM byte stream")
    parser.add_argument("--out-dir", default="vllm_omni_out")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.stream and args.mode != "single":
        raise SystemExit("--stream is only valid with --mode single")
    ref_audio = encode_audio(args.ref_audio)
    if args.mode == "single":
        run_single(args, ref_audio)
    else:
        run_batch(args, ref_audio)


if __name__ == "__main__":
    main()
