#!/usr/bin/env python3
"""
Concatenate all WAV files in a directory into a single long WAV file
"""
import argparse
import os
from pathlib import Path
import torchaudio
import torch


def concatenate_wav_files(input_dir, output_path, sort_by="name", silence_duration=0.5):
    """
    Concatenate all WAV files in input directory into a single file
    
    Args:
        input_dir: Directory containing WAV files
        output_path: Path for output concatenated file
        sort_by: Sorting method - "name", "modified_time", or "none"
        silence_duration: Duration of silence (in seconds) to add between audio files
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    
    # Find all WAV files
    wav_files = list(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return False
    
    # Sort files based on chosen method
    if sort_by == "name":
        wav_files.sort(key=lambda x: x.name.lower())
    elif sort_by == "modified_time":
        wav_files.sort(key=lambda x: x.stat().st_mtime)
    # If sort_by is "none", keep original order (glob may be unsorted)
    
    print(f"Found {len(wav_files)} WAV files:")
    for i, wav_file in enumerate(wav_files):
        print(f"  {i+1:3d}. {wav_file.name}")
    
    # Load first file to get sample rate and channels
    first_audio, sample_rate = torchaudio.load(str(wav_files[0]))
    num_channels = first_audio.shape[0]
    
    print(f"\nSample rate: {sample_rate}Hz")
    print(f"Channels: {num_channels}")
    print(f"Duration of first file: {first_audio.shape[1]/sample_rate:.2f}s")
    
    # Calculate silence tensor
    silence_samples = int(silence_duration * sample_rate)
    silence_tensor = torch.zeros((num_channels, silence_samples))
    
    # Start with first audio
    concatenated_audio = first_audio
    total_duration = first_audio.shape[1] / sample_rate
    
    for wav_file in wav_files[1:]:
        audio, sr = torchaudio.load(str(wav_file))
        
        # Verify same sample rate and channels
        if sr != sample_rate:
            print(f"Warning: {wav_file.name} has different sample rate ({sr} vs {sample_rate}), skipping...")
            continue
        if audio.shape[0] != num_channels:
            print(f"Warning: {wav_file.name} has different channel count ({audio.shape[0]} vs {num_channels}), skipping...")
            continue
        
        # Add silence between current concatenated audio and new audio
        concatenated_audio = torch.cat([concatenated_audio, silence_tensor, audio], dim=1)
        duration = audio.shape[1] / sr
        total_duration += duration + silence_duration  # Include silence in total duration
        print(f"Added {wav_file.name} ({duration:.2f}s) + {silence_duration}s silence, total: {total_duration:.2f}s")
    
    # Save concatenated audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), concatenated_audio, sample_rate)
    
    print(f"\nSuccessfully saved concatenated audio to: {output_path}")
    print(f"Total duration: {total_duration:.2f}s ({int(total_duration//60):d}:{total_duration%60:05.2f})")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate all WAV files in a directory into one long audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir", 
        type=str, 
        required=True,
        help="Directory containing WAV files to concatenate"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        required=True,
        help="Output path for concatenated WAV file"
    )
    parser.add_argument(
        "--sort-by", 
        type=str, 
        choices=["name", "modified_time", "none"], 
        default="name",
        help="How to sort the files before concatenating"
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=1,
        help="Duration of silence (in seconds) to insert between audio files"
    )
    
    args = parser.parse_args()
    
    success = concatenate_wav_files(
        input_dir=args.input_dir,
        output_path=args.output_path,
        sort_by=args.sort_by,
        silence_duration=args.silence_duration
    )
    
    if success:
        print(f"\nConcatenation completed successfully!")
    else:
        print(f"\nFailed to concatenate files.")
        exit(1)