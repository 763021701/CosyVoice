#!/usr/bin/env python3
"""
Batch compose dialogue WAV files

Usage:
    python tools/batch_compose_dialogues.py \
        --input-dir outputs/Admission_outputs \
        --output-dir outputs/Admission_outputs/composed \
        --silence-duration 0.5
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compose_wav import concatenate_wav_files


def batch_compose_dialogues(
    input_dir: str,
    output_dir: str = None,
    sort_by: str = "name",
    silence_duration: float = 0.5,
    pattern: str = "dialogue_*",
    quiet: bool = False
):
    """
    Batch compose all dialogue directories into individual WAV files.
    
    Args:
        input_dir: Parent directory containing dialogue_xxxx subdirectories
        output_dir: Output directory for composed files (default: input_dir/composed)
        sort_by: Sorting method for WAV files within each dialogue
        silence_duration: Silence duration between utterances
        pattern: Glob pattern to match dialogue directories
        quiet: Suppress detailed output, only show progress
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return False
    
    # Default output directory
    if output_dir is None:
        output_dir = input_dir / "composed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all dialogue directories
    dialogue_dirs = sorted([d for d in input_dir.glob(pattern) if d.is_dir()])
    
    if not dialogue_dirs:
        print(f"[ERROR] No dialogue directories found matching '{pattern}' in {input_dir}")
        return False
    
    print(f"[INFO] Found {len(dialogue_dirs)} dialogue directories, output: {output_dir}")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for idx, dialogue_dir in enumerate(dialogue_dirs, 1):
        dialogue_id = dialogue_dir.name
        output_path = output_dir / f"{dialogue_id}.wav"
        
        # Skip if output already exists
        if output_path.exists():
            skipped_count += 1
            success_count += 1
            if not quiet:
                print(f"[{idx}/{len(dialogue_dirs)}] {dialogue_id} - SKIP (exists)")
            continue
        
        # Check if directory has WAV files
        wav_files = list(dialogue_dir.glob("*.wav"))
        if not wav_files:
            if not quiet:
                print(f"[{idx}/{len(dialogue_dirs)}] {dialogue_id} - SKIP (no WAV files)")
            continue
        
        try:
            result = concatenate_wav_files(
                input_dir=dialogue_dir,
                output_path=output_path,
                sort_by=sort_by,
                silence_duration=silence_duration,
                verbose=not quiet
            )
            
            if result:
                success_count += 1
                print(f"[{idx}/{len(dialogue_dirs)}] {dialogue_id} - OK ({len(wav_files)} files)")
            else:
                failed_count += 1
                print(f"[{idx}/{len(dialogue_dirs)}] {dialogue_id} - FAILED")
        
        except Exception as e:
            failed_count += 1
            print(f"[{idx}/{len(dialogue_dirs)}] {dialogue_id} - ERROR: {e}")
    
    # Summary
    print(f"[DONE] Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")
    
    return failed_count == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch compose dialogue WAV files from synthesize_dialogue.py output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Parent directory containing dialogue_xxxx subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for composed files (default: input_dir/composed)"
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["name", "modified_time", "none"],
        default="name",
        help="How to sort WAV files within each dialogue"
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.5,
        help="Duration of silence (seconds) between utterances"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="dialogue_*",
        help="Glob pattern to match dialogue directories"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output, only show progress"
    )
    
    args = parser.parse_args()
    
    success = batch_compose_dialogues(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sort_by=args.sort_by,
        silence_duration=args.silence_duration,
        pattern=args.pattern,
        quiet=args.quiet
    )
    
    if not success:
        exit(1)
