from glob import glob
import os
import argparse

from tqdm import tqdm

from modules.pseudo_sensing import determine_sensing_points
from modules.utils import load_samples_json

def run(sample, verbose=False):
    """
    Run pseudo sensing on a sample.
    
    Args:
    sample: dict, sample data
    """
    video_dir = f"sample_video/cropped"
    for i, sensor in enumerate(sample['sensors']):
        video = os.path.join(video_dir, f"{sample['id']}/{i}.mp4")
        DURATION_SECONDS = 20
        output_name = f"{sample['id']}_{i}_{sensor['movement']['type']}_{sensor['movement']['direction']}"
        determine_sensing_points(
            video, 
            output_name, 
            sample['info']['frame_rate'], 
            sensor['start_frame'], 
            DURATION_SECONDS, 
            filter_size=(4, 4), 
            vobose_visualization=verbose
        )

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sample_id", type=str, default=None, help="ID of the sample to preprocess")
    arg_parser.add_argument("--all", action="store_true", help="Preprocess all samples")
    args = arg_parser.parse_args()

    samples = load_samples_json()
    if args.sample_id:
        sample = next((s for s in samples if s["id"] == args.sample_id), None)
        if sample is None:
            raise ValueError(f"Sample with ID {args.sample_id} not found")
        print(f"Processing Sample {sample['id']}")
        run(sample, verbose=False)
    elif args.all:
        for sample in tqdm(samples, desc="Processing Samples"):
            run(sample, verbose=False)
    else:
        arg_parser.print_help()