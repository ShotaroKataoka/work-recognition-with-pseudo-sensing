import os
import argparse

from tqdm import tqdm

from pseudo_sensing.modules.crop_regions import crop_regions
from modules.utils import load_samples_json

def preprocess_sample(sample):
    video_path = f"sample_video/{sample['filename']}"
    regions = [(s['x'], s['y'], s['w'], s['h']) for s in sample['sensers']]
    output_files = [f"sample_video/cropped/{sample['id']}/{i}.mp4" for i in range(len(regions))]

    os.makedirs(f"sample_video/cropped/{sample['id']}", exist_ok=True)

    crop_regions(video_path, regions, output_files)


def preprocess_all_samples(samples):
    for sample in tqdm(samples, desc="Processing Samples"):
        preprocess_sample(sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", type=str, default=None, help="ID of the sample to preprocess")
    parser.add_argument("--all", action="store_true", help="Preprocess all samples")
    args = parser.parse_args()
    
    samples = load_samples_json()

    if args.sample_id:
        sample = next((s for s in samples if s["id"] == args.sample_id), None)
        if sample is None:
            raise ValueError(f"Sample with ID {args.sample_id} not found")
        preprocess_sample(sample)
    elif args.all:
        preprocess_all_samples(samples)
    else:
        parser.print_help()

        