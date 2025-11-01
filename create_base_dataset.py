from datasets import load_dataset, Audio
import sys

# Get speaker parameter from command line or use default
speaker = sys.argv[1] if len(sys.argv) > 1 else "Linda"

ds = load_dataset("csv", data_files="/data/1-audio-chunks/Linda/manifest.csv")["train"]
ds = ds.cast_column("audio", Audio(sampling_rate=48000))
# Optional split
# ds = ds.train_test_split(test_size=0.05, seed=42)
# save as HuggingFace dataset (arrow) folder
ds.save_to_disk(f"/data/2-base-datasets/{speaker}")
# or push
# ds.push_to_hub(f"hf_repo/dataset_{speaker}")