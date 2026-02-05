#!/usr/bin/env python3
"""
医疗场景对话语音数据合成工具

用途：
- 读取对话文本文件（格式：[角色] 说话内容）
- 自动识别对话中的角色并分配不同音色
- 使用 CosyVoice 合成每句对话
- 支持多进程并行、断点续传

支持的场景：
1. 门诊对话：医生、患者
2. 手术记录：主刀医生、麻醉医生、器械护士、一助医生
3. 查房问诊：主治医生、患者、住院医生（可选）、实习生（可选）
4. 其他场景：自动识别 [角色名] 格式的任意角色

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


# 角色名称标准化映射表
# 支持多种称呼方式映射到统一的角色标识符
ROLE_ALIASES = {
    # 门诊场景
    "doctor": "doctor",
    "医生": "doctor",
    "医师": "doctor",
    "patient": "patient",
    "患者": "patient",
    "病人": "patient",
    
    # 手术场景
    "主刀医生": "chief_surgeon",
    "主刀": "chief_surgeon",
    "chief_surgeon": "chief_surgeon",
    "麻醉医生": "anesthesiologist",
    "麻醉师": "anesthesiologist",
    "anesthesiologist": "anesthesiologist",
    "器械护士": "instrument_nurse",
    "器械": "instrument_nurse",
    "instrument_nurse": "instrument_nurse",
    "一助医生": "first_assistant",
    "一助": "first_assistant",
    "first_assistant": "first_assistant",
    "二助医生": "second_assistant",
    "二助": "second_assistant",
    "second_assistant": "second_assistant",
    
    # 查房场景
    "主治医生": "attending_doctor",
    "主治": "attending_doctor",
    "attending_doctor": "attending_doctor",
    "住院医生": "resident_doctor",
    "住院医": "resident_doctor",
    "resident_doctor": "resident_doctor",
    "实习生": "intern",
    "实习医生": "intern",
    "intern": "intern",
    
    # 护理场景
    "护士": "nurse",
    "护士长": "head_nurse",
    "nurse": "nurse",
}

# 角色显示名称（用于日志输出）
ROLE_DISPLAY_NAMES = {
    "doctor": "医生",
    "patient": "患者",
    "chief_surgeon": "主刀医生",
    "anesthesiologist": "麻醉医生",
    "instrument_nurse": "器械护士",
    "first_assistant": "一助医生",
    "second_assistant": "二助医生",
    "attending_doctor": "主治医生",
    "resident_doctor": "住院医生",
    "intern": "实习生",
    "nurse": "护士",
    "head_nurse": "护士长",
}


def normalize_role(role_text: str) -> Optional[str]:
    """
    将角色文本标准化为统一的角色标识符。
    
    Args:
        role_text: 原始角色文本，如 "主刀医生"、"医生"、"患者" 等
    
    Returns:
        标准化后的角色标识符，如 "chief_surgeon"、"doctor"、"patient"
        如果无法识别，返回 None
    """
    role_text = role_text.strip().lower()
    
    # 1. 直接匹配
    if role_text in ROLE_ALIASES:
        return ROLE_ALIASES[role_text]
    
    # 2. 部分匹配（按优先级排序，优先匹配更具体的角色）
    priority_patterns = [
        # 手术场景（优先匹配，因为包含"医生"等通用词）
        ("主刀", "chief_surgeon"),
        ("麻醉", "anesthesiologist"),
        ("器械", "instrument_nurse"),
        ("一助", "first_assistant"),
        ("二助", "second_assistant"),
        # 查房场景
        ("主治", "attending_doctor"),
        ("住院医", "resident_doctor"),
        ("实习", "intern"),
        # 护理场景
        ("护士长", "head_nurse"),
        ("护士", "nurse"),
        # 门诊场景（最后匹配，作为兜底）
        ("医生", "doctor"),
        ("医师", "doctor"),
        ("患者", "patient"),
        ("病人", "patient"),
    ]
    
    for pattern, role_id in priority_patterns:
        if pattern in role_text:
            return role_id
    
    # 3. 无法识别的角色 - 使用原始文本作为标识符（转为小写，去空格）
    # 这样可以支持任意自定义角色
    sanitized = re.sub(r'\s+', '_', role_text.strip())
    if sanitized:
        return sanitized
    
    return None


def parse_dialogue_file(filepath: Path) -> List[Dict[str, str]]:
    """
    解析对话文本文件。
    
    支持多种场景的角色格式：
    
    门诊对话示例：
        [医生] 您好，请问哪里不舒服？
        [患者] 我最近总是头痛。
    
    手术记录示例：
        [主刀医生] 麻醉医生，患者生命体征如何？
        [麻醉医生] 血压125/80，血氧98%，生命体征平稳。
        [器械护士] 结肠镜已准备就绪。
        [一助医生] 主刀，患者EMR记录显示有慢性浅表性胃炎病史。
    
    查房问诊示例：
        [主治医生] 今天感觉怎么样？
        [患者] 比昨天好多了。
        [住院医生] 体温已经正常了。
        [实习生] 血常规结果也恢复正常了。
    
    返回：[{"role": "chief_surgeon", "text": "...", "role_display": "主刀医生"}, ...]
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
        
        # 标准化角色
        role = normalize_role(role_text)
        if not role:
            print(f"[WARNING] Unrecognized role: '{role_text}', skipping line")
            continue
        
        # 获取显示名称
        role_display = ROLE_DISPLAY_NAMES.get(role, role_text)
        
        utterances.append({
            "role": role,
            "role_display": role_display,
            "text": content
        })
    
    # 调试：打印解析结果
    print(f"[DEBUG] parse_dialogue_file: parsed {len(utterances)} utterances")
    for i, utt in enumerate(utterances[:5]):
        text_preview = utt['text'][:30] + '...' if len(utt['text']) > 30 else utt['text']
        print(f"        [{i}] role='{utt['role']}' ({utt['role_display']}) text='{text_preview}'")
    
    return utterances


def extract_unique_roles(utterances: List[Dict[str, str]]) -> List[str]:
    """
    从对话句子中提取所有唯一的角色，保持出现顺序。
    
    Args:
        utterances: 解析后的对话句子列表
    
    Returns:
        唯一角色列表，按首次出现顺序排列
    """
    seen = set()
    unique_roles = []
    for utt in utterances:
        role = utt["role"]
        if role not in seen:
            seen.add(role)
            unique_roles.append(role)
    return unique_roles


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
            print(f"[WARNING] Failed to read text file: {txt_file.name} - {e}")
    
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
        num_speakers: 需要选择的说话人数量（根据对话中实际角色数量动态确定）
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
        utterances: 对话句子列表 [{"role": "chief_surgeon", "text": "...", "role_display": "主刀医生"}]
        speaker_mapping: 角色到音频文件的映射（支持任意数量的角色）
                         例如：{"chief_surgeon": Path(...), "anesthesiologist": Path(...), ...}
        speaker_prompt_mapping: 角色到完整 prompt 文本的映射
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
    
    # Debug: print speaker_mapping
    print(f"[DEBUG] synthesize_dialogue received speaker_mapping:")
    for role_key, spk_path in speaker_mapping.items():
        print(f"        {role_key}: {spk_path} (exists: {spk_path.exists() if hasattr(spk_path, 'exists') else 'N/A'})")
    
    # 为每个角色缓存 prompt（避免重复跑 ONNX）
    cached_prompts = {}
    
    for idx, utt in enumerate(utterances):
        role = utt["role"]
        text = utt["text"]
        speaker_wav = speaker_mapping.get(role)
        
        print(f"[DEBUG] Processing utterance {idx}, role={role}, speaker_wav={speaker_wav}")
        
        if not speaker_wav or not speaker_wav.exists():
            print(f"[WARNING] Role {role} has no speaker file, skipping: {text[:20]}...")
            print(f"           speaker_wav={speaker_wav}, exists={speaker_wav.exists() if speaker_wav else 'None'}")
            continue
        
        # 输出文件命名：{dialogue_id}_{idx:03d}_{role}.wav
        output_file = output_dir / f"{dialogue_id}_{idx:03d}_{role}.wav"
        
        # 断点续传：如果已存在且非空，跳过
        if output_file.exists() and output_file.stat().st_size > 1000:
            print(f"[SKIP] Already exists: {output_file.name}")
            generated_files.append(output_file)
            continue
        
        # 如果该角色的 prompt 还未缓存，先缓存
        if role not in cached_prompts:
            try:
                # 获取该角色对应的 prompt 文本
                role_prompt_text = speaker_prompt_mapping.get(role)
                if not role_prompt_text:
                    print(f"[ERROR] Role {role} has no prompt text, skipping")
                    continue
                print(f"[INFO] Loading {role} speaker voice: {speaker_wav.name}")
                print(f"       prompt text: {role_prompt_text[:50]}...")
                
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
                print(f"[INFO] {role} speaker voice loaded successfully")
            
            except Exception as e:
                print(f"[ERROR] Failed to load {role} speaker voice: {speaker_wav.name}")
                print(f"         Error details: {e}")
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
            print(f"[ERROR] Synthesis failed: {output_file.name}")
            print(f"         Role: {role}, Text: {text[:30]}...")
            print(f"         Error details: {e}")
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
    parser.add_argument("--num-workers", type=int, default=1, help="总进程数（用于数据分片）")
    
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
    print(f"[INFO] Loading model: {args.model_dir}")
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
    
    print(f"[INFO] Model loaded: {cosyvoice.__class__.__name__}")
    
    # 扫描对话文件
    dialogue_files = sorted(dialogue_dir.glob(args.dialogue_pattern))
    if not dialogue_files:
        raise ValueError(f"对话目录中没有找到匹配的文件：{dialogue_dir / args.dialogue_pattern}")
    
    # 扫描音色文件
    speaker_files = sorted(speaker_dir.glob(args.speaker_pattern))
    if not speaker_files:
        raise ValueError(f"音色库目录中没有找到匹配的文件：{speaker_dir / args.speaker_pattern}")
    
    if len(speaker_files) < 2:
        raise ValueError(f"Speaker files count ({len(speaker_files)}) is less than 2, cannot assign voices")
    
    print(f"[INFO] Dialogue files count: {len(dialogue_files)}")
    print(f"[INFO] Speaker files count: {len(speaker_files)}")
    
    # 验证音色文件（快速检查前几个）
    print(f"[INFO] Validating speaker files...")
    invalid_speakers = []
    for spk_file in speaker_files[:min(5, len(speaker_files))]:
        try:
            wav, sr = torchaudio.load(str(spk_file))
            duration = wav.shape[1] / sr
            if duration > 30:
                print(f"[WARNING] Speaker file exceeds 30s limit: {spk_file.name} ({duration:.2f}s)")
                print(f"           Recommend trimming to 5-8s for best results")
                invalid_speakers.append(spk_file)
            elif duration < 2:
                print(f"[WARNING] Speaker file too short: {spk_file.name} ({duration:.2f}s)")
                print(f"           Recommend using audio longer than 3s")
        except Exception as e:
            print(f"[WARNING] Cannot read speaker file: {spk_file.name} - {e}")
            invalid_speakers.append(spk_file)
    
    if invalid_speakers:
        print(f"[WARNING] Found {len(invalid_speakers)} problematic speaker files, synthesis may fail")
    
    # 多进程分片：当前进程只处理属于自己的对话
    worker_id = args.worker_id
    num_workers = args.num_workers
    my_dialogues = [f for i, f in enumerate(dialogue_files) if i % num_workers == worker_id]
    
    print(f"[INFO] Worker {worker_id}/{num_workers}, handling {len(my_dialogues)} dialogues")
    
    # 记录统计
    total_dialogues = len(my_dialogues)
    total_utterances = 0
    total_success = 0
    
    # 逐个对话处理
    for idx, dialogue_file in enumerate(my_dialogues, 1):
        dialogue_id = dialogue_file.stem  # 文件名（不含扩展名）
        print(f"\n[{idx}/{total_dialogues}] Processing dialogue: {dialogue_id}")
        
        try:
            # 解析对话
            utterances = parse_dialogue_file(dialogue_file)
            if not utterances:
                print(f"[WARNING] Dialogue file is empty or incorrectly formatted, skipping: {dialogue_file.name}")
                continue
            
            print(f"  - Total utterances: {len(utterances)}")
            total_utterances += len(utterances)
            
            # 提取对话中所有唯一角色
            unique_roles = extract_unique_roles(utterances)
            num_roles = len(unique_roles)
            
            if num_roles == 0:
                print(f"[WARNING] No valid roles found in dialogue, skipping: {dialogue_file.name}")
                continue
            
            # 检查音色文件数是否足够
            if len(speaker_files) < num_roles:
                print(f"[WARNING] Not enough speaker files ({len(speaker_files)}) for {num_roles} roles, skipping: {dialogue_file.name}")
                continue
            
            # 打印解析后的对话结构（调试用）
            print(f"  - Unique roles ({num_roles}): {unique_roles}")
            for i, utt in enumerate(utterances[:5]):
                text_preview = utt['text'][:30] + '...' if len(utt['text']) > 30 else utt['text']
                print(f"    [{i}] {utt['role_display']}: {text_preview}")
            
            # 为该对话随机分配音色（根据实际角色数量动态分配）
            # 使用 dialogue_id 作为种子，保证每次运行分配相同音色
            dialogue_seed = args.seed + hash(dialogue_id) % 100000
            selected_speakers = select_speakers(speaker_files, num_speakers=num_roles, seed=dialogue_seed)
            
            # 动态构建角色到音色的映射
            speaker_mapping = {}
            for role, speaker_path in zip(unique_roles, selected_speakers):
                speaker_mapping[role] = speaker_path
            
            # 打印音色分配信息
            print(f"  - Speaker assignment:")
            for role, spk_path in speaker_mapping.items():
                role_display = ROLE_DISPLAY_NAMES.get(role, role)
                print(f"    {role_display} ({role}): {spk_path.name}")
            # 为每个角色加载对应的 prompt 文本
            speaker_prompt_mapping = {}
            for role, spk_wav in speaker_mapping.items():
                prompt_text = load_speaker_prompt_text(spk_wav, args.prompt_prefix, args.default_content)
                speaker_prompt_mapping[role] = prompt_text
                role_display = ROLE_DISPLAY_NAMES.get(role, role)
                # print(f"    {role_display} prompt: {prompt_text[:50]}...")
            
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
            print(f"  - Successfully synthesized: {len(generated)} audio files")
            
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
            print(f"[ERROR] Failed to process dialogue: {dialogue_file.name} - {e}")
            import traceback
            traceback.print_exc()
    
    # Final statistics
    print("\n" + "=" * 60)
    print(f"[DONE] Worker {worker_id}/{num_workers}")
    print(f"  - Dialogues processed: {total_dialogues}")
    print(f"  - Total utterances: {total_utterances}")
    print(f"  - Successfully synthesized: {total_success} audio files")
    print(f"  - Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
