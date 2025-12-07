from datasets import load_dataset, Audio
import sys

# Get speaker and sampling_rate parameters from command line (mandatory)
if len(sys.argv) < 3:
    print("Error: Speaker and sampling_rate parameters are required")
    print("Usage: python create_base_dataset.py <speaker> <sampling_rate>")
    sys.exit(1)

speaker = sys.argv[1]
sampling_rate = int(sys.argv[2])

ds = load_dataset("csv", data_files=f"/data/1-audio-chunks/{speaker}/manifest.csv")["train"]
ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
# Optional split
# ds = ds.train_test_split(test_size=0.05, seed=42)
# save as HuggingFace dataset (arrow) folder
ds.save_to_disk(f"/data/2-base-datasets/{speaker}")
# or push
# ds.push_to_hub(f"hf_repo/dataset_{speaker}")