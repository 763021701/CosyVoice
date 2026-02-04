#!/usr/bin/env python3
"""
医患对话语音数据合成工具

用途：
- 读取医患对话文本文件（格式：[医生/患者] 说话内容）
- 随机分配两个不同音色作为医生和患者
- 使用 CosyVoice 合成每句对话
- 支持多进程并行、断点续传

示例用法：
    # 单进程合成
    python tools/synthesize_dialogue.py \\
        --model-dir pretrained_models/Fun-CosyVoice3-0.5B \\
        --dialogue-dir data/dialogues \\
        --speaker-dir data/speakers \\
        --output-dir outputs/synthesized \\
        --concurrency 2 \\
        --fp16

    # 多进程并行（推荐）
    python tools/synthesize_dialogue.py ... --worker-id 0 --num-workers 6 &
    python tools/synthesize_dialogue.py ... --worker-id 1 --num-workers 6 &
    ...
    python tools/synthesize_dialogue.py ... --worker-id 5 --num-workers 6 &
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio


def parse_dialogue_file(filepath: Path) -> List[Dict[str, str]]:
    """
    解析对话文本文件。
    
    格式示例：
        [医生] 您好，请问哪里不舒服？
        [患者] 我最近总是头痛。
        [医生] 头痛持续多久了？
        [患者] 大概三天了。
    
    返回：[{"role": "doctor", "text": "您好..."}, {"role": "patient", "text": "我最近..."}]
    """
    lines = filepath.read_text(encoding="utf-8").strip().splitlines()
    utterances = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 匹配 [角色] 文本 格式
        match = re.match(r"\[([^\]]+)\]\s*(.+)", line)
        if not match:
            continue
        
        role_text, content = match.groups()
        role_text = role_text.strip()
        content = content.strip()
        
        # 识别角色
        if "医生" in role_text or "医师" in role_text or "doctor" in role_text.lower():
            role = "doctor"
        elif "患者" in role_text or "病人" in role_text or "patient" in role_text.lower():
            role = "patient"
        else:
            # 未识别的角色，跳过
            continue
        
        utterances.append({"role": role, "text": content})
    
    # 调试：打印解析结果
    print(f"[DEBUG] parse_dialogue_file: 解析了 {len(utterances)} 句话")
    for i, utt in enumerate(utterances):
        print(f"        [{i}] role='{utt['role']}' text='{utt['text'][:30]}...'")
    
    return utterances


def load_speaker_prompt_text(speaker_wav: Path, prompt_prefix: str, default_content: str = "") -> str:
    """
    加载音色文件对应的 prompt 文本。
    
    查找与音色文件同名的 .txt 文件（例如 speaker_001.wav 对应 speaker_001.txt），
    读取其中的文本内容，然后拼接成完整的 prompt_text。
    
    例如：
        - speaker_001.txt 内容：希望你以后能够做的比我还好呦。
        - prompt_prefix：You are a helpful assistant.<|endofprompt|>
        - 返回：You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。
    
    Args:
        speaker_wav: 音色文件路径
        prompt_prefix: prompt 前缀（通常包含 <|endofprompt|> 标记）
        default_content: 如果找不到文本文件，使用的默认内容
    
    Returns:
        完整的 prompt 文本
    """
    txt_file = speaker_wav.with_suffix('.txt')
    content = default_content
    
    if txt_file.exists():
        try:
            file_content = txt_file.read_text(encoding='utf-8').strip()
            if file_content:
                content = file_content
        except Exception as e:
            print(f"[WARNING] 读取文本文件失败：{txt_file.name} - {e}")
    
    # 拼接完整的 prompt_text
    return prompt_prefix + content


def select_speakers(
    speaker_files: List[Path],
    num_speakers: int = 2,
    seed: Optional[int] = None
) -> List[Path]:
    """
    随机选择 N 个不同的说话人音频文件。
    
    Args:
        speaker_files: 所有可用的说话人音频文件列表
        num_speakers: 需要选择的说话人数量（通常是2：医生+患者）
        seed: 随机种子（用于可复现）
    
    Returns:
        选中的说话人文件列表
    """
    if len(speaker_files) < num_speakers:
        raise ValueError(f"音色库文件数 ({len(speaker_files)}) 少于所需数量 ({num_speakers})")
    
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    return rng.sample(speaker_files, num_speakers)


def synthesize_dialogue(
    cosyvoice,
    utterances: List[Dict[str, str]],
    speaker_mapping: Dict[str, Path],
    speaker_prompt_mapping: Dict[str, str],
    output_dir: Path,
    dialogue_id: str,
    mode: str = "zero_shot",
    speed: float = 1.0,
    sample_rate: int = 22050,
) -> List[Path]:
    """
    合成一个对话中的所有句子。
    
    Args:
        cosyvoice: CosyVoice 模型实例
        utterances: 对话句子列表 [{"role": "doctor", "text": "..."}]
        speaker_mapping: 角色到音频文件的映射 {"doctor": Path(...), "patient": Path(...)}
        speaker_prompt_mapping: 角色到完整 prompt 文本的映射 {"doctor": "prefix+content", "patient": "prefix+content"}
        output_dir: 输出目录
        dialogue_id: 对话ID（用于文件命名）
        mode: 合成模式
        speed: 语速倍率
        sample_rate: 采样率
    
    Returns:
        生成的音频文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # 调试：打印 speaker_mapping
    print(f"[DEBUG] synthesize_dialogue 收到的 speaker_mapping:")
    for role_key, spk_path in speaker_mapping.items():
        print(f"        {role_key}: {spk_path} (exists: {spk_path.exists() if hasattr(spk_path, 'exists') else 'N/A'})")
    
    # 为每个角色缓存 prompt（避免重复跑 ONNX）
    cached_prompts = {}
    
    for idx, utt in enumerate(utterances):
        role = utt["role"]
        text = utt["text"]
        speaker_wav = speaker_mapping.get(role)
        
        print(f"[DEBUG] 处理第 {idx} 句，角色={role}, speaker_wav={speaker_wav}")
        
        if not speaker_wav or not speaker_wav.exists():
            print(f"[WARNING] 角色 {role} 没有对应的音色文件，跳过：{text[:20]}...")
            print(f"           speaker_wav={speaker_wav}, exists={speaker_wav.exists() if speaker_wav else 'None'}")
            continue
        
        # 输出文件命名：{dialogue_id}_{idx:03d}_{role}.wav
        output_file = output_dir / f"{dialogue_id}_{idx:03d}_{role}.wav"
        
        # 断点续传：如果已存在且非空，跳过
        if output_file.exists() and output_file.stat().st_size > 1000:
            print(f"[SKIP] 已存在：{output_file.name}")
            generated_files.append(output_file)
            continue
        
        # 如果该角色的 prompt 还未缓存，先缓存
        if role not in cached_prompts:
            try:
                # 获取该角色对应的 prompt 文本
                role_prompt_text = speaker_prompt_mapping.get(role)
                if not role_prompt_text:
                    print(f"[ERROR] 角色 {role} 没有对应的 prompt 文本，跳过")
                    continue
                print(f"[INFO] 加载 {role} 音色：{speaker_wav.name}")
                print(f"       prompt文本：{role_prompt_text[:50]}...")
                
                if mode == "zero_shot":
                    cached_prompts[role] = cosyvoice.frontend.frontend_zero_shot(
                        "", role_prompt_text, str(speaker_wav), sample_rate, ""
                    )
                elif mode == "cross_lingual":
                    cached_prompts[role] = cosyvoice.frontend.frontend_cross_lingual(
                        "", str(speaker_wav), sample_rate, ""
                    )
                else:
                    raise ValueError(f"不支持的 mode: {mode}")
                
                # 清掉空 text（后面每条都会覆盖）
                cached_prompts[role].pop("text", None)
                cached_prompts[role].pop("text_len", None)
                print(f"[INFO] {role} 音色加载成功")
            
            except Exception as e:
                print(f"[ERROR] 加载 {role} 音色失败：{speaker_wav.name}")
                print(f"         错误详情：{e}")
                import traceback
                traceback.print_exc()
                # 跳过该角色的所有句子
                continue
        
        # 合成当前句子
        try:
            # 准备 model_input
            model_input = dict(cached_prompts[role])
            txt_token, txt_len = cosyvoice.frontend._extract_text_token(text)
            model_input["text"] = txt_token
            model_input["text_len"] = txt_len
            
            # 调用模型
            for out in cosyvoice.model.tts(**model_input, stream=False, speed=speed):
                wav = out["tts_speech"]  # shape: [1, T]
                torchaudio.save(str(output_file), wav, sample_rate)
                print(f"[OK] {output_file.name} ({wav.shape[1] / sample_rate:.2f}s)")
                generated_files.append(output_file)
                break  # stream=False 只会 yield 一次
        
        except Exception as e:
            print(f"[ERROR] 合成失败：{output_file.name}")
            print(f"         角色：{role}, 文本：{text[:30]}...")
            print(f"         错误详情：{e}")
            import traceback
            traceback.print_exc()
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="医患对话语音数据合成工具（支持多进程并行）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 模型参数
    parser.add_argument("--model-dir", type=str, required=True, help="CosyVoice 模型目录")
    parser.add_argument("--fp16", action="store_true", help="使用 fp16 加速")
    parser.add_argument("--load-vllm", action="store_true", help="启用 vLLM（如果支持）")
    parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "cross_lingual"], help="合成模式")
    parser.add_argument("--prompt-prefix", type=str, default="You are a helpful assistant.<|endofprompt|>", help="prompt 前缀（会自动拼接音色文件对应的文本内容）")
    parser.add_argument("--default-content", type=str, default="希望你以后能够做的比我还好呦。", help="默认文本内容（如果音色文件没有对应的 .txt 文件）")
    parser.add_argument("--speed", type=float, default=1, help="语速倍率")
    parser.add_argument("--concurrency", type=int, default=2, help="单进程内并发数（推荐 1~2）")
    
    # 数据路径
    parser.add_argument("--dialogue-dir", type=str, required=True, help="对话文本目录（每个文件一个对话）")
    parser.add_argument("--speaker-dir", type=str, required=True, help="说话人音色库目录（.wav 文件）")
    parser.add_argument("--output-dir", type=str, default="outputs/synthesized", help="输出目录")
    
    # 多进程参数
    parser.add_argument("--worker-id", type=int, default=0, help="当前进程编号（0-based）")
    parser.add_argument("--num-workers", type=int, default=6, help="总进程数（用于数据分片）")
    
    # 随机种子
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于可复现的音色分配）")
    
    # 文件格式
    parser.add_argument("--dialogue-pattern", type=str, default="*.txt", help="对话文件匹配模式（glob）")
    parser.add_argument("--speaker-pattern", type=str, default="*.wav", help="音色文件匹配模式（glob）")
    
    args = parser.parse_args()
    
    # 设置路径
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.append(str(repo_root / "third_party" / "Matcha-TTS"))
    
    from cosyvoice.cli.cosyvoice import AutoModel
    
    dialogue_dir = Path(args.dialogue_dir)
    speaker_dir = Path(args.speaker_dir)
    output_dir = Path(args.output_dir)
    
    if not dialogue_dir.exists():
        raise FileNotFoundError(f"对话目录不存在：{dialogue_dir}")
    if not speaker_dir.exists():
        raise FileNotFoundError(f"音色库目录不存在：{speaker_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"[INFO] 加载模型：{args.model_dir}")
    model_kwargs = {
        "model_dir": args.model_dir,
        "fp16": args.fp16,
        "load_vllm": args.load_vllm,
        "load_trt": False,
        "trt_concurrent": 1,
        "load_jit": False,
    }
    # 兼容不同模型的参数
    while True:
        try:
            cosyvoice = AutoModel(**model_kwargs)
            break
        except TypeError as e:
            key = None
            if "unexpected keyword argument" in str(e) and "'" in str(e):
                key = str(e).split("'")[1]
            if key and key in model_kwargs:
                model_kwargs.pop(key)
                continue
            raise
    
    # 禁用 CUDA cleanup（提升并行性能）
    os.environ["COSYVOICE_DISABLE_CUDA_CLEANUP"] = "1"
    try:
        cosyvoice.model.disable_cuda_cleanup = True
    except:
        pass
    
    print(f"[INFO] 模型加载完成：{cosyvoice.__class__.__name__}")
    
    # 扫描对话文件
    dialogue_files = sorted(dialogue_dir.glob(args.dialogue_pattern))
    if not dialogue_files:
        raise ValueError(f"对话目录中没有找到匹配的文件：{dialogue_dir / args.dialogue_pattern}")
    
    # 扫描音色文件
    speaker_files = sorted(speaker_dir.glob(args.speaker_pattern))
    if not speaker_files:
        raise ValueError(f"音色库目录中没有找到匹配的文件：{speaker_dir / args.speaker_pattern}")
    
    if len(speaker_files) < 2:
        raise ValueError(f"音色库文件数少于 2，无法分配医生和患者音色")
    
    print(f"[INFO] 对话文件数：{len(dialogue_files)}")
    print(f"[INFO] 音色文件数：{len(speaker_files)}")
    
    # 验证音色文件（快速检查前几个）
    print(f"[INFO] 验证音色文件...")
    invalid_speakers = []
    for spk_file in speaker_files[:min(5, len(speaker_files))]:
        try:
            wav, sr = torchaudio.load(str(spk_file))
            duration = wav.shape[1] / sr
            if duration > 30:
                print(f"[WARNING] 音色文件超过30秒限制：{spk_file.name} ({duration:.2f}s)")
                print(f"           建议裁剪到5-8秒以获得最佳效果")
                invalid_speakers.append(spk_file)
            elif duration < 2:
                print(f"[WARNING] 音色文件过短：{spk_file.name} ({duration:.2f}s)")
                print(f"           建议使用3秒以上的音频")
        except Exception as e:
            print(f"[WARNING] 音色文件无法读取：{spk_file.name} - {e}")
            invalid_speakers.append(spk_file)
    
    if invalid_speakers:
        print(f"[WARNING] 发现 {len(invalid_speakers)} 个问题音色文件，合成时可能出错")
    
    # 多进程分片：当前进程只处理属于自己的对话
    worker_id = args.worker_id
    num_workers = args.num_workers
    my_dialogues = [f for i, f in enumerate(dialogue_files) if i % num_workers == worker_id]
    
    print(f"[INFO] 进程 {worker_id}/{num_workers}，负责 {len(my_dialogues)} 个对话")
    
    # 记录统计
    total_dialogues = len(my_dialogues)
    total_utterances = 0
    total_success = 0
    
    # 逐个对话处理
    for idx, dialogue_file in enumerate(my_dialogues, 1):
        dialogue_id = dialogue_file.stem  # 文件名（不含扩展名）
        print(f"\n[{idx}/{total_dialogues}] 处理对话：{dialogue_id}")
        
        try:
            # 解析对话
            utterances = parse_dialogue_file(dialogue_file)
            if not utterances:
                print(f"[WARNING] 对话文件为空或格式不正确，跳过：{dialogue_file.name}")
                continue
            
            print(f"  - 共 {len(utterances)} 句对话")
            total_utterances += len(utterances)
            
            # 打印解析后的对话结构（调试用）
            for i, utt in enumerate(utterances):
                print(f"    [{i}] {utt['role']}: {utt['text'][:30]}...")
            
            # 为该对话随机分配两个音色（医生、患者）
            # 使用 dialogue_id 作为种子，保证每次运行分配相同音色
            dialogue_seed = args.seed + hash(dialogue_id) % 100000
            selected_speakers = select_speakers(speaker_files, num_speakers=2, seed=dialogue_seed)
            
            speaker_mapping = {
                "doctor": selected_speakers[0],
                "patient": selected_speakers[1],
            }
            
            print(f"  - 医生音色：{selected_speakers[0].name} (存在: {selected_speakers[0].exists()})")
            print(f"  - 患者音色：{selected_speakers[1].name} (存在: {selected_speakers[1].exists()})")
            print(f"  - speaker_mapping: {speaker_mapping}")
            
            # 为每个角色加载对应的 prompt 文本
            speaker_prompt_mapping = {}
            for role, spk_wav in speaker_mapping.items():
                prompt_text = load_speaker_prompt_text(spk_wav, args.prompt_prefix, args.default_content)
                speaker_prompt_mapping[role] = prompt_text
                print(f"  - {role} prompt文本: {prompt_text[:60]}...")
            
            # 合成对话
            generated = synthesize_dialogue(
                cosyvoice=cosyvoice,
                utterances=utterances,
                speaker_mapping=speaker_mapping,
                speaker_prompt_mapping=speaker_prompt_mapping,
                output_dir=output_dir / dialogue_id,  # 每个对话一个子目录
                dialogue_id=dialogue_id,
                mode=args.mode,
                speed=args.speed,
                sample_rate=cosyvoice.sample_rate,
            )
            
            total_success += len(generated)
            print(f"  - 成功合成：{len(generated)} 个音频文件")
            
            # 保存元数据（方便后续追溯）
            meta_file = output_dir / dialogue_id / f"{dialogue_id}_meta.json"
            meta = {
                "dialogue_id": dialogue_id,
                "source_file": str(dialogue_file),
                "speaker_mapping": {k: str(v) for k, v in speaker_mapping.items()},
                "speaker_prompt_mapping": speaker_prompt_mapping,
                "utterances": utterances,
                "generated_files": [str(f) for f in generated],
            }
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        
        except Exception as e:
            print(f"[ERROR] 处理对话失败：{dialogue_file.name} - {e}")
            import traceback
            traceback.print_exc()
    
    # 最终统计
    print("\n" + "=" * 60)
    print(f"[完成] 进程 {worker_id}/{num_workers}")
    print(f"  - 处理对话数：{total_dialogues}")
    print(f"  - 总句子数：{total_utterances}")
    print(f"  - 成功合成：{total_success} 个音频")
    print(f"  - 输出目录：{output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
