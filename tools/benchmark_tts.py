import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class BenchResult:
    total_wall_s: float
    total_audio_s: float
    utterances: int
    max_mem_allocated_bytes: int
    max_mem_reserved_bytes: int

    @property
    def rtf(self) -> float:
        # Real Time Factor = wall / audio (越小越快)
        if self.total_audio_s <= 0:
            return float("inf")
        return self.total_wall_s / self.total_audio_s

    @property
    def audio_seconds_per_second(self) -> float:
        # 吞吐：每秒合成多少秒音频（越大越好）
        if self.total_wall_s <= 0:
            return 0.0
        return self.total_audio_s / self.total_wall_s


def _read_texts(text: Optional[str], text_file: Optional[str], n: int) -> List[str]:
    if text_file:
        p = Path(text_file)
        if not p.exists():
            raise FileNotFoundError(f"text_file not found: {text_file}")
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            raise ValueError(f"text_file is empty: {text_file}")
        if n > 0:
            # 循环复用，保证可以测到更稳定的吞吐
            return [lines[i % len(lines)] for i in range(n)]
        return lines
    if text is None:
        text = "你好，我是通义生成式语音大模型。为了测试吞吐与显存占用，我正在进行批量语音合成评估。"
    if n <= 0:
        n = 32
    return [text for _ in range(n)]


def _ensure_endofprompt_for_cosyvoice3(prompt_text: str) -> str:
    # CosyVoice3 推荐在 prompt 里包含 <|endofprompt|>
    if "<|endofprompt|>" in prompt_text:
        return prompt_text
    return prompt_text.rstrip() + "<|endofprompt|>"


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def _synthesize_one(
    cosyvoice,
    mode: str,
    tts_text: str,
    *,
    spk_id: str,
    prompt_text: str,
    instruct_text: str,
    prompt_wav: str,
    cached_prompt: Optional[Dict],
    stream: bool,
    speed: float,
    text_frontend: bool,
) -> Tuple[float, int]:
    """
    返回：(audio_seconds, num_segments)
    """
    total_audio_s = 0.0
    segs = 0

    # 自己做文本切分，避免 inference_* 里 tqdm 多线程刷屏
    segments = cosyvoice.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)
    for seg in segments:
        if mode == "sft":
            model_input = cosyvoice.frontend.frontend_sft(seg, spk_id)
        elif mode == "zero_shot":
            if cached_prompt is None:
                model_input = cosyvoice.frontend.frontend_zero_shot(
                    seg, prompt_text, prompt_wav, cosyvoice.sample_rate, ""
                )
            else:
                # 复用 prompt 相关的 token/embedding/feat，只更新 text
                model_input = dict(cached_prompt)
                txt, txt_len = cosyvoice.frontend._extract_text_token(seg)
                model_input["text"] = txt
                model_input["text_len"] = txt_len
        elif mode == "cross_lingual":
            if cached_prompt is None:
                model_input = cosyvoice.frontend.frontend_cross_lingual(
                    seg, prompt_wav, cosyvoice.sample_rate, ""
                )
            else:
                model_input = dict(cached_prompt)
                txt, txt_len = cosyvoice.frontend._extract_text_token(seg)
                model_input["text"] = txt
                model_input["text_len"] = txt_len
        elif mode == "instruct2":
            if cached_prompt is None:
                model_input = cosyvoice.frontend.frontend_instruct2(
                    seg, instruct_text, prompt_wav, cosyvoice.sample_rate, ""
                )
            else:
                model_input = dict(cached_prompt)
                txt, txt_len = cosyvoice.frontend._extract_text_token(seg)
                model_input["text"] = txt
                model_input["text_len"] = txt_len
        else:
            raise ValueError(f"unsupported mode: {mode}")

        # cosyvoice.model.tts(...) 会 yield 1 次（stream=False）或多次（stream=True）
        for out in cosyvoice.model.tts(**model_input, stream=stream, speed=speed):
            wav = out["tts_speech"]
            # wav shape: [1, T]
            total_audio_s += wav.shape[1] / cosyvoice.sample_rate
        segs += 1

    return total_audio_s, segs


def run_benchmark(
    cosyvoice,
    texts: List[str],
    *,
    mode: str,
    concurrency: int,
    warmup: int,
    spk_id: str,
    prompt_text: str,
    instruct_text: str,
    prompt_wav: str,
    cache_prompt: bool,
    stream: bool,
    speed: float,
    text_frontend: bool,
) -> BenchResult:
    # prompt 预处理缓存：避免每条都跑 ONNX/whisper/kaldi
    cached_prompt: Optional[Dict] = None
    if mode in {"zero_shot", "cross_lingual", "instruct2"} and cache_prompt:
        if mode == "zero_shot":
            cached_prompt = cosyvoice.frontend.frontend_zero_shot(
                "", prompt_text, prompt_wav, cosyvoice.sample_rate, ""
            )
        elif mode == "cross_lingual":
            cached_prompt = cosyvoice.frontend.frontend_cross_lingual(
                "", prompt_wav, cosyvoice.sample_rate, ""
            )
        elif mode == "instruct2":
            cached_prompt = cosyvoice.frontend.frontend_instruct2(
                "", instruct_text, prompt_wav, cosyvoice.sample_rate, ""
            )
        # 清掉空 text（后面每条都会覆盖）
        cached_prompt.pop("text", None)
        cached_prompt.pop("text_len", None)

    # warmup（可让 CUDA/ONNX 初始化完成，数据更稳定）
    for i in range(max(0, warmup)):
        _synthesize_one(
            cosyvoice,
            mode,
            texts[i % len(texts)],
            spk_id=spk_id,
            prompt_text=prompt_text,
            instruct_text=instruct_text,
            prompt_wav=prompt_wav,
            cached_prompt=cached_prompt,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend,
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    total_audio_s = 0.0

    # 多线程并发：对单卡通常比多进程更友好（避免权重重复占显存）
    import concurrent.futures as cf

    def _job(t: str) -> Tuple[float, int]:
        return _synthesize_one(
            cosyvoice,
            mode,
            t,
            spk_id=spk_id,
            prompt_text=prompt_text,
            instruct_text=instruct_text,
            prompt_wav=prompt_wav,
            cached_prompt=cached_prompt,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend,
        )

    if concurrency <= 1:
        for t in texts:
            a_s, _ = _job(t)
            total_audio_s += a_s
    else:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_job, t) for t in texts]
            for fut in cf.as_completed(futures):
                a_s, _ = fut.result()
                total_audio_s += a_s

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    max_alloc = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    max_resv = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0

    return BenchResult(
        total_wall_s=(t1 - t0),
        total_audio_s=float(total_audio_s),
        utterances=len(texts),
        max_mem_allocated_bytes=max_alloc,
        max_mem_reserved_bytes=max_resv,
    )


def main():
    parser = argparse.ArgumentParser(
        description="CosyVoice2/3 语音合成资源评估（RTF/吞吐/显存峰值），支持并发线程与 prompt 缓存。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=str, required=True, help="模型目录或 ModelScope 名称")
    parser.add_argument(
        "--mode",
        type=str,
        default="zero_shot",
        choices=["sft", "zero_shot", "cross_lingual", "instruct2"],
        help="合成模式",
    )
    parser.add_argument("--fp16", action="store_true", help="加载 fp16（通常更快、显存更省）")
    parser.add_argument("--load-vllm", action="store_true", help="仅对 CosyVoice2/3 生效：启用 vLLM（需已安装 vllm）")
    parser.add_argument("--concurrency", type=int, default=1, help="线程并发数（同进程共享权重显存）")
    parser.add_argument("--warmup", type=int, default=2, help="warmup 次数（不计入统计）")
    parser.add_argument("--n", type=int, default=32, help="合成条数（若 text_file 行数不足会循环复用）")
    parser.add_argument("--text", type=str, default=None, help="单条文本（会重复 n 次）")
    parser.add_argument("--text-file", type=str, default=None, help="文本文件（utf-8，一行一条）")
    parser.add_argument("--text-frontend", action="store_true", help="启用文本前端（TN+分句）。关闭可更贴近官方 demo 复现。")

    parser.add_argument("--spk-id", type=str, default="中文女", help="SFT 模式说话人（仅 mode=sft 用）")
    parser.add_argument("--prompt-wav", type=str, default="asset/zero_shot_prompt.wav", help="zero-shot/instruct2/cross-lingual 的提示音频")
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="希望你以后能够做的比我还好呦。",
        help="zero-shot 的提示文本（CosyVoice3 建议包含 <|endofprompt|>）",
    )
    parser.add_argument(
        "--instruct-text",
        type=str,
        default="You are a helpful assistant. 请用四川话说这句话<|endofprompt|>",
        help="instruct2 的提示文本（建议包含 <|endofprompt|>）",
    )
    parser.add_argument(
        "--cache-prompt",
        dest="cache_prompt",
        action="store_true",
        default=True,
        help="缓存 prompt 预处理（强烈建议，避免重复跑 ONNX/whisper/kaldi）",
    )
    parser.add_argument(
        "--no-cache-prompt",
        dest="cache_prompt",
        action="store_false",
        help="禁用 prompt 缓存（用于测全链路：包含每条都做 prompt 预处理）",
    )
    parser.add_argument(
        "--enable-cuda-cleanup",
        dest="disable_cuda_cleanup",
        action="store_false",
        default=True,
        help="启用每条推理后的 CUDA 清理/同步（更保守，但吞吐更慢；默认禁用以获得更好并行性能）",
    )
    parser.add_argument("--stream", action="store_true", help="流式合成（更像实时场景，会多次 yield）")
    parser.add_argument("--speed", type=float, default=1.0, help="语速倍率（非流式更稳定）")

    args = parser.parse_args()

    import sys

    # 确保从任意工作目录运行，都能 import 本仓库的 cosyvoice 包
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # 保持与仓库示例一致：third_party/Matcha-TTS
    sys.path.append(str(repo_root / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel

    texts = _read_texts(args.text, args.text_file, args.n)
    prompt_wav = str(Path(args.prompt_wav))
    if args.mode in {"zero_shot", "cross_lingual", "instruct2"}:
        if not Path(prompt_wav).exists():
            raise FileNotFoundError(f"prompt_wav not found: {prompt_wav}")

    # 兼容 CosyVoice / CosyVoice2 / CosyVoice3 不同的 __init__ 参数集合：
    # - CosyVoice: (model_dir, load_jit, load_trt, fp16, trt_concurrent)
    # - CosyVoice2: (model_dir, load_jit, load_trt, load_vllm, fp16, trt_concurrent)
    # - CosyVoice3: (model_dir, load_trt, load_vllm, fp16, trt_concurrent)
    model_kwargs = {
        "model_dir": args.model_dir,
        "fp16": bool(args.fp16),
        "load_trt": False,
        "trt_concurrent": 1,
        "load_vllm": bool(args.load_vllm),
        "load_jit": False,
    }
    # 自动剔除不被目标模型支持的参数（避免 TypeError: unexpected keyword argument）
    while True:
        try:
            cosyvoice = AutoModel(**model_kwargs)
            break
        except TypeError as e:
            msg = str(e)
            key = None
            # Python 常见报错格式：got an unexpected keyword argument 'xxx'
            if "unexpected keyword argument" in msg and "'" in msg:
                key = msg.split("'")[1]
            if key and key in model_kwargs:
                model_kwargs.pop(key)
                continue
            raise

    model_name = cosyvoice.__class__.__name__

    # 禁用 cosyvoice/cli/model.py 中每条推理后的 empty_cache + synchronize（对吞吐影响很大）
    if bool(args.disable_cuda_cleanup):
        os.environ["COSYVOICE_DISABLE_CUDA_CLEANUP"] = "1"
        try:
            cosyvoice.model.disable_cuda_cleanup = True
        except Exception:
            pass

    prompt_text = args.prompt_text
    instruct_text = args.instruct_text
    if model_name == "CosyVoice3":
        # CosyVoice3 文本建议含 <|endofprompt|>
        if args.mode == "zero_shot":
            prompt_text = _ensure_endofprompt_for_cosyvoice3(prompt_text)
        if args.mode == "instruct2":
            instruct_text = _ensure_endofprompt_for_cosyvoice3(instruct_text)

    # 跑基准
    res = run_benchmark(
        cosyvoice,
        texts,
        mode=args.mode,
        concurrency=int(args.concurrency),
        warmup=int(args.warmup),
        spk_id=args.spk_id,
        prompt_text=prompt_text,
        instruct_text=instruct_text,
        prompt_wav=prompt_wav,
        cache_prompt=bool(args.cache_prompt),
        stream=bool(args.stream),
        speed=float(args.speed),
        text_frontend=bool(args.text_frontend),
    )

    # 输出
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==== CosyVoice Benchmark ====")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"fp16: {bool(args.fp16)}")
    print(f"load_vllm: {bool(args.load_vllm)}")
    print(f"concurrency: {int(args.concurrency)}")
    print(f"stream: {bool(args.stream)}")
    print(f"text_frontend: {bool(args.text_frontend)}")
    print(f"utterances: {res.utterances}")
    print(f"total_audio_s: {res.total_audio_s:.3f}")
    print(f"total_wall_s: {res.total_wall_s:.3f}")
    print(f"RTF (wall/audio): {res.rtf:.4f}  (越小越快)")
    print(f"Throughput (audio_s/s): {res.audio_seconds_per_second:.3f}  (越大越好)")
    if torch.cuda.is_available():
        print(f"GPU peak allocated: {_format_bytes(res.max_mem_allocated_bytes)}")
        print(f"GPU peak reserved:  {_format_bytes(res.max_mem_reserved_bytes)}")
    print("=============================")


if __name__ == "__main__":
    # 避免某些环境下多线程 tokenization 过度占用 CPU
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

