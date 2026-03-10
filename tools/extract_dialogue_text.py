#!/usr/bin/env python3
"""
Extract clean text from dialogue files for ASR evaluation (ground truth).

Removes:
1. Role markers at line start: [医生], [患者], [主治医生], etc.
2. Paralinguistic markers:
   - [breath], [sigh], [quick_breath], etc. (English in brackets)
   - <laughter>...</laughter>, etc. (XML-like tags)

Usage:
    # Process single file
    python tools/extract_dialogue_text.py \
        --input-file syn_data/Admission_dialogues/dialogue_0001.txt \
        --output-file outputs/dialogue_0001_text.txt

    # Batch process directory
    python tools/extract_dialogue_text.py \
        --input-dir syn_data/Admission_dialogues \
        --output-dir outputs/Admission_texts \
        --pattern "dialogue_*.txt"
"""
import argparse
import re
from pathlib import Path


def clean_dialogue_text(text: str) -> str:
    """
    Clean dialogue text by removing role markers and paralinguistic markers.
    
    Args:
        text: Raw dialogue text (multi-line)
    
    Returns:
        Cleaned text as a single paragraph
    """
    lines = text.strip().splitlines()
    cleaned_parts = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 1. Remove role marker at line start: [中文角色名] 
        # Match: [任意非英文字符] at the beginning
        line = re.sub(r'^\[[^\]]*[^\x00-\x7F][^\]]*\]\s*', '', line)
        
        # 2. Remove paralinguistic markers with English content in brackets: [breath], [sigh], etc.
        line = re.sub(r'\[[a-zA-Z_]+\]', '', line)
        
        # 3. Remove XML-like tags with English: <laughter>...</laughter>, <cough>, etc.
        # Remove paired tags: <tag>content</tag> -> content
        line = re.sub(r'<([a-zA-Z_]+)>(.*?)</\1>', r'\2', line)
        # Remove self-closing or unpaired tags: <tag>, </tag>
        line = re.sub(r'</?[a-zA-Z_]+>', '', line)
        
        # 4. Clean up extra whitespace
        line = re.sub(r'\s+', ' ', line).strip()
        
        if line:
            cleaned_parts.append(line)
    
    # Join all lines into a single paragraph
    return ' '.join(cleaned_parts)


def process_single_file(input_file: Path, output_file: Path) -> bool:
    """Process a single dialogue file."""
    try:
        text = input_file.read_text(encoding='utf-8')
        cleaned = clean_dialogue_text(text)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(cleaned, encoding='utf-8')
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process {input_file.name}: {e}")
        return False


def batch_process(
    input_dir: str,
    output_dir: str,
    pattern: str = "dialogue_*.txt",
    quiet: bool = False
):
    """
    Batch process dialogue files.
    
    Args:
        input_dir: Directory containing dialogue text files
        output_dir: Output directory for cleaned text files
        pattern: Glob pattern to match dialogue files
        quiet: Suppress detailed output
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    input_files = sorted(input_dir.glob(pattern))
    
    if not input_files:
        print(f"[ERROR] No files found matching '{pattern}' in {input_dir}")
        return False
    
    print(f"[INFO] Found {len(input_files)} files, output: {output_dir}")
    
    success_count = 0
    failed_count = 0
    
    for idx, input_file in enumerate(input_files, 1):
        output_file = output_dir / input_file.name
        
        if process_single_file(input_file, output_file):
            success_count += 1
            if not quiet:
                print(f"[{idx}/{len(input_files)}] {input_file.name} - OK")
        else:
            failed_count += 1
            print(f"[{idx}/{len(input_files)}] {input_file.name} - FAILED")
    
    print(f"[DONE] Success: {success_count}, Failed: {failed_count}")
    return failed_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract clean text from dialogue files for ASR evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Single file mode
    parser.add_argument(
        "--input-file",
        type=str,
        help="Single input dialogue file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for cleaned text"
    )
    
    # Batch mode
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing dialogue text files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for cleaned text files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="dialogue_*.txt",
        help="Glob pattern to match dialogue files"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.input_file:
        # Single file mode
        input_file = Path(args.input_file)
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_file = input_file.with_suffix('.clean.txt')
        
        if process_single_file(input_file, output_file):
            print(f"[OK] Saved to: {output_file}")
            
            # Preview
            cleaned = output_file.read_text(encoding='utf-8')
            print(f"\n--- Preview (first 200 chars) ---")
            print(cleaned[:200] + "..." if len(cleaned) > 200 else cleaned)
        else:
            exit(1)
    
    elif args.input_dir:
        # Batch mode
        if not args.output_dir:
            print("[ERROR] --output-dir is required for batch mode")
            exit(1)
        
        success = batch_process(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            quiet=args.quiet
        )
        if not success:
            exit(1)
    
    else:
        parser.print_help()
        print("\n[ERROR] Either --input-file or --input-dir is required")
        exit(1)


if __name__ == "__main__":
    main()
