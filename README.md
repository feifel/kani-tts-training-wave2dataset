# SpeechSlicer – End-to-End Dataset Builder for TTS / ASR

Turn long-form speech recordings into a ready-to-use HuggingFace dataset in minutes.

## What it does
1. Detects speaker segments automatically (Silero-VAD + WhisperX diarization)
2. Cuts long audio into clean, speaker-specific, 1-5s clips
3. Writes a manifest CSV (`audio_path,text,duration`)
4. Converts the manifest into a HF dataset (Arrow) with 48 kHz audio column
5. Publishes the dataset locally

## Quick start
```bash
# 1. install
pip install -r requirements.txt

# 2. drop long audio files into ./audio_in/
# 3. run pipeline
python slice_pipeline.py

# 4. create HF dataset for speaker "linda"
python create_hf_dataset.py linda
```

## Pipeline steps
- slice_pipeline.py	full VAD → diarization → slicing → manifest
- create_hf_dataset.py <speaker>	manifest → HF Dataset → hf_repo/dataset_<speaker>

## Output layout
```plaintext
out/
├── linda_0001.wav
├── …
└── manifest.csv   # audio_path,text,duration

hf_repo/
└── dataset_linda/ # Arrow dataset, 48 kHz
```
## Customisation
- Change target sample-rate in create_hf_dataset.py
- Adjust min-segment / padding seconds inside slice_pipeline.py
- Swap Whisper model size via --model large-v2

## Requirements
Python ≥3.8, CUDA (for GPU diarization)
Core dependencies: faster-whisper, whisperx, silero-vad, datasets, torchaudio

## License
MIT – feel free to fork & PR.