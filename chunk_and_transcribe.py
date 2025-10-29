#!/usr/bin/env python3
"""
chunk_and_transcribe.py

- Detect silence regions in audio
- Split audio on silent passages
- Recursively split long chunks until they are under a max duration
- Trim silence from chunk edges
- Transcribe chunks with faster-whisper
- Save results to CSV/JSONL

Usage:
    python chunk_and_transcribe.py -i long_audio.wav -o out_dir --silence_duration 0.5 --max_chunk_duration 5.0
"""

import os
import re
import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from tqdm import tqdm
import pandas as pd
import subprocess   # add at the top with the other imports
import math

def samples_from_seconds(seconds: float, sr: int) -> int:
    """Convert seconds to number of samples."""
    return int(round(seconds * sr))

def ensure_audio_mono(path: str):
    """Read audio file and convert it to mono float32."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

def detect_silence_regions(audio: np.ndarray, sr: int, threshold: float = 0.01, min_silence_duration: float = 0.5) -> List[Tuple[int, int]]:
    """Detect silence regions in audio that are longer than min_silence_duration."""
    min_silence_samples = samples_from_seconds(min_silence_duration, sr)
    silence_regions = []
    
    is_silent = np.abs(audio) < threshold
    in_silence = False
    silence_start = 0
    
    for i in range(len(is_silent)):
        if is_silent[i] and not in_silence:
            silence_start = i
            in_silence = True
        elif not is_silent[i] and in_silence:
            silence_end = i
            if (silence_end - silence_start) >= min_silence_samples:
                silence_regions.append((silence_start, silence_end))
            in_silence = False
    
    if in_silence and (len(is_silent) - silence_start) >= min_silence_samples:
        silence_regions.append((silence_start, len(is_silent)))
    
    return silence_regions

def create_speech_segments(audio_length: int, silence_regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Create speech segments by taking the inverse of silence regions."""
    if not silence_regions:
        return [(0, audio_length)]
    
    silence_regions = sorted(silence_regions, key=lambda x: x[0])
    
    segments = []
    current_start = 0
    
    for silence_start, silence_end in silence_regions:
        if silence_start > current_start:
            segments.append((current_start, silence_start))
        current_start = silence_end
    
    if current_start < audio_length:
        segments.append((current_start, audio_length))
    
    return segments

def recursively_split_chunk(
    segment: Tuple[int, int], 
    silence_regions: List[Tuple[int, int]], 
    sr: int, 
    max_len_seconds: float = 5.0,
    min_split_silence_duration: float = 0.1  # ← NEW
) -> List[Tuple[int, int]]:
    """
    Recursively splits a segment until all resulting chunks are shorter than max_len_seconds.
    Splits are made at the longest silence within the segment, even if shorter than 0.5s.
    """
    start, end = segment
    max_len_samples = samples_from_seconds(max_len_seconds, sr)
    min_split_silence_samples = samples_from_seconds(min_split_silence_duration, sr)

    # Base case: segment is already short enough
    if (end - start) <= max_len_samples:
        return [(start, end)]

    # Find all silences strictly inside the current segment
    internal_silences = []
    for s_start, s_end in silence_regions:
        if s_start > start and s_end < end:
            duration = s_end - s_start
            if duration >= min_split_silence_samples:  # ← RELAXED
                internal_silences.append(((s_start, s_end), duration))

    if not internal_silences:
        print(f"Warning: Segment from {start/sr:.2f}s to {end/sr:.2f}s is longer than {max_len_seconds}s but contains no internal silence ≥{min_split_silence_duration}s to split on. Keeping it as one chunk.")
        return [(start, end)]

    # Find the longest internal silence to split on
    longest_silence, _ = max(internal_silences, key=lambda x: x[1])
    s_start, s_end = longest_silence

    # Split the segment into two parts around the silence
    part1 = (start, s_start)
    part2 = (s_end, end)

    # Recurse on both parts and combine the results
    result_chunks = []
    result_chunks.extend(recursively_split_chunk(part1, silence_regions, sr, max_len_seconds, min_split_silence_duration))
    result_chunks.extend(recursively_split_chunk(part2, silence_regions, sr, max_len_seconds, min_split_silence_duration))
    
    return result_chunks

def trim_silence_edges(segments: List[Tuple[int, int]], audio: np.ndarray, sr: int, 
                      edge_silence_padding: float = 0.1, threshold: float = 0.01) -> List[Tuple[int, int]]:
    """
    Reduces silence at the start and end of each segment to a specified padding duration.
    """
    padding_samples = samples_from_seconds(edge_silence_padding, sr)
    result_segments = []

    for start, end in segments:
        chunk = audio[start:end]
        is_silent = np.abs(chunk) < threshold
        
        # Find first non-silent sample
        first_sound = np.argmax(is_silent == False)
        
        # Find last non-silent sample
        last_sound = len(chunk) - 1 - np.argmax(np.flip(is_silent) == False)

        new_start = start + max(0, first_sound - padding_samples)
        new_end = start + min(len(chunk), last_sound + padding_samples)

        if new_end > new_start:
            result_segments.append((new_start, new_end))
            
    return result_segments

def filter_short_segments(segments: List[Tuple[int, int]], min_len_samples: int) -> List[Tuple[int, int]]:
    """Remove segments that are too short."""
    return [seg for seg in segments if (seg[1] - seg[0]) >= min_len_samples]

def main():
    parser = argparse.ArgumentParser(description="Split a long audio into chunks using a recursive silence-based algorithm and transcribe with faster-whisper.")
    parser.add_argument("-i", "--input", required=True, help="Input audio file (long)")
    parser.add_argument("-o", "--out_dir", required=True, help="Output directory")
    parser.add_argument("--min_dur", type=float, default=0.2, help="Minimum chunk duration in seconds (default 0.2)")
    parser.add_argument("--max_chunk_duration", type=float, default=5.0, help="Maximum chunk duration in seconds. Chunks longer than this will be recursively split. (default 5.0)")
    parser.add_argument("--silence_threshold", type=float, default=0.01, help="Amplitude threshold for silence detection (default 0.01)")
    parser.add_argument("--silence_duration", type=float, default=0.5, help="Minimum duration for a passage to be considered silence in seconds (default 0.5s)")
    parser.add_argument("--internal_silence_duration", type=float, default=0.1, help="Minimum silence duration (in seconds) that the recursive splitter is allowed to use for splitting. Should be ≤ --silence_duration. (default 0.1)")
    parser.add_argument("--edge_silence_padding", type=float, default=0.1, help="Seconds of silence to keep at the start and end of chunks (default 0.1s)")
    parser.add_argument("--model", type=str, default="large-v3", help="faster-whisper model id (default large-v3)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for models: cuda or cpu (default cuda)")
    parser.add_argument("--format", choices=["csv","jsonl"], default="csv", help="Output manifest format")
    parser.add_argument("--language", type=str, default=None, help="Language code for transcription (e.g. 'en', 'es', 'fr'). Auto-detect if not specified.")
    parser.add_argument("--speaker", type=str, default="unknown", help="Speaker name to embed in the manifest (default: unknown)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading audio...")
    original_audio, original_sr = ensure_audio_mono(args.input)
    audio_length = len(original_audio)
    print(f"Audio length: {audio_length / original_sr:.2f} seconds")

    # 1. Detect all silences > 500ms (as per user request)
    print(f"Detecting silence regions > {args.silence_duration}s...")
    silence_regions = detect_silence_regions(
        original_audio, original_sr, 
        threshold=args.silence_threshold,
        min_silence_duration=args.silence_duration
    )
    print(f"Found {len(silence_regions)} silence regions.")

    print("Detecting all silence regions ≥ {:.2f} s...".format(args.internal_silence_duration))
    all_silence_regions = detect_silence_regions(
        original_audio, original_sr,
        threshold=args.silence_threshold,
        min_silence_duration=args.internal_silence_duration      # ← NEW
    )

    print("Detecting initial-split silence regions ≥ {:.2f} s...".format(args.silence_duration))
    initial_silence_regions = [s for s in all_silence_regions
                            if (s[1] - s[0]) >= samples_from_seconds(args.silence_duration, original_sr)]

    # 2. Create initial speech segments by splitting on all detected silences
    print("Creating initial speech segments...")
    initial_segments = create_speech_segments(audio_length, initial_silence_regions)
    print(f"Created {len(initial_segments)} initial segments.")

    # 3. Recursively split any segments that are too long
    print(f"Recursively splitting segments longer than {args.max_chunk_duration}s...")
    split_segments = []
    for segment in tqdm(initial_segments, desc="Splitting long segments"):
        chunks = recursively_split_chunk(
            segment, all_silence_regions, original_sr,   # ← pass the *full* list here
            max_len_seconds=args.max_chunk_duration
        )
        split_segments.extend(chunks)
    print(f"Total segments after recursive splitting: {len(split_segments)}")

    # 4. Trim silence from the edges of each chunk
    print(f"Trimming silence from segment edges to {args.edge_silence_padding}s...")
    trimmed_segments = trim_silence_edges(
        split_segments, original_audio, original_sr,
        edge_silence_padding=args.edge_silence_padding,
        threshold=args.silence_threshold
    )
    print(f"Segments after trimming: {len(trimmed_segments)}")

    # 5. Filter out any segments that are now too short
    min_len_samples = samples_from_seconds(args.min_dur, original_sr)
    final_segments = filter_short_segments(trimmed_segments, min_len_samples)
    print(f"Final chunks after filtering short segments: {len(final_segments)}")

    # 6. Sort final segments by start time and transcribe
    final_segments = sorted(final_segments, key=lambda x: x[0])
    
    print(f"Loading transcription model: {args.model}")
    compute_type = "float16" if args.device == "cuda" and torch.cuda.is_available() else "int8"
    wmodel = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    records = []
    for idx, (samp_s, samp_e) in enumerate(tqdm(final_segments, desc="Transcribing chunks")):
        start_sec = samp_s / original_sr
        end_sec   = samp_e / original_sr
        wav_out   = str(out_dir / f"chunk_{idx:05d}_{start_sec:.3f}_{end_sec:.3f}.wav")

        chunk_wav = original_audio[samp_s:samp_e]
        sf.write(wav_out, chunk_wav, original_sr, subtype='PCM_16')

        # ── RMS-based background filter ───────────────────────────────
        stat = subprocess.check_output(
            ["sox", wav_out, "-n", "stat"], stderr=subprocess.STDOUT, text=True
        )
        m = re.search(r'RMS.*?:\s*([0-9.]+)', stat)
        rms_linear = float(m.group(1)) if m else 0.0
        rms_db = -float('inf') if rms_linear == 0.0 else 20 * math.log10(rms_linear)

        if rms_db > -30.0:                                       # keep
            pass
        else:                                                    # too quiet
            print(f"Removed: {wav_out}  RMS = {rms_db:.2f} dBFS")# visible log
            os.remove(wav_out)
            continue
        # ──────────────────────────────────────────────────────────────

        # Transcribe and build manifest exactly as before
        segments_gen, _ = wmodel.transcribe(wav_out, beam_size=5, language=args.language)
        transcript = " ".join(seg.text.strip() for seg in segments_gen)
        if not transcript.strip():
            os.remove(wav_out)
            continue

        records.append({
            "audio"      : wav_out,
            "text"       : transcript,
            "speaker"    : args.speaker,         
            "language"   : args.language or "auto"
        })

    # Write manifest file
    out_manifest = out_dir / ("manifest." + args.format)
    if args.format == "csv":
        pd.DataFrame(records).to_csv(out_manifest, index=False)
    else: # jsonl
        with open(out_manifest, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Manifest saved to: {out_manifest}")
    print(f"{len(records)} WAV chunks saved to: {out_dir}")

if __name__ == "__main__":
    main()