from datasets import load_dataset, Audio
import sys

# Get speaker parameter from command line or use default
speaker = sys.argv[1].lower() if len(sys.argv) > 1 else "linda"

ds = load_dataset("csv", data_files="out/manifest.csv")["train"]
ds = ds.cast_column("audio", Audio(sampling_rate=48000))
# Optional split
# ds = ds.train_test_split(test_size=0.05, seed=42)
# save as HuggingFace dataset (arrow) folder
ds.save_to_disk(f"hf_repo/dataset_{speaker}")
# or push
# ds.push_to_hub(f"hf_repo/dataset_{speaker}")