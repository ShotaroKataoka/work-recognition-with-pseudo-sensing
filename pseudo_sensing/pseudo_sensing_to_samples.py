import os
import argparse

from tqdm import tqdm

from modules.pseudo_sensing import determine_sensing_points
from modules.utils import load_samples_json

def run(
        sample,
        save_grayscale=False,
        save_filtered=False,
        save_waveforms=False,
        save_wavelets=False
    ):
    """
    Run pseudo sensing on a sample.
    
    Args:
    sample: dict, sample data
    save_grayscale: bool, save the grayscale video
    save_filtered: bool, save the filtered video
    save_waveforms: bool, save the waveforms
    save_wavelets: bool, save the wave
    """
    video_dir = f"sample_video/cropped"
    for i, sensor in enumerate(sample['sensors']):
        video = os.path.join(video_dir, f"{sample['id']}/{i}.mp4")
        DURATION_SECONDS = 60
        output_name = f"{sample['id']}_{i}_{sensor['movement']['type']}_{sensor['movement']['direction']}"
        print(f"Processing Sensor {i} of Sample {sample['id']}")
        determine_sensing_points(
            video, 
            output_name, 
            sample['info']['frame_rate'], 
            sensor['start_frame'], 
            DURATION_SECONDS, 
            filter_size=(4, 4), 
            save_grayscale=save_grayscale,
            save_filtered=save_filtered,
            save_waveforms=save_waveforms,
            save_wavelets=save_wavelets
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
        run(sample, save_wavelets=False)
    elif args.all:
        for sample in tqdm(samples, desc="Processing Samples"):
            run(sample)
    else:
        arg_parser.print_help()