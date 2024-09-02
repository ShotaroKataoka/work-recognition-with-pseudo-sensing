import os
import argparse

from tqdm import tqdm

from modules.pseudo_sensing import determine_sensing_points
from modules.utils import load_samples_json

def run(
        sample,
        args
    ):
    """
    Run pseudo sensing on a sample.
    
    Args:
    sample: dict, sample data
    args: argparse.Namespace, arguments
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
            save_all=args.save_all,
            save_grayscale=args.save_grayscale,
            save_filtered=args.save_filtered,
            save_waveforms=args.save_waveforms,
            save_wavelets=args.save_wavelets,
            save_cluster_grid=args.save_cluster_grid,
            save_cluster_scater=args.save_cluster_scater
        )
    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sample_id", type=str, default=None, help="ID of the sample to preprocess")
    arg_parser.add_argument("--all", action="store_true", help="Preprocess all samples")
    arg_parser.add_argument("--save_all", action="store_true", help="Save all the outputs")
    arg_parser.add_argument("--save_grayscale", action="store_true", help="Save the grayscale video")
    arg_parser.add_argument("--save_filtered", action="store_true", help="Save the filtered video")
    arg_parser.add_argument("--save_waveforms", action="store_true", help="Save the waveforms")
    arg_parser.add_argument("--save_wavelets", action="store_true", help="Save the wavelets")
    arg_parser.add_argument("--save_cluster_grid", action="store_true", help="Save the cluster grid")
    arg_parser.add_argument("--save_cluster_scater", action="store_true", help="Save the cluster scatter plot")
    args = arg_parser.parse_args()

    samples = load_samples_json()
    if args.sample_id:
        sample = next((s for s in samples if s["id"] == args.sample_id), None)
        if sample is None:
            raise ValueError(f"Sample with ID {args.sample_id} not found")
        print(f"Processing Sample {sample['id']}")
        run(sample, args)
    elif args.all:
        for sample in tqdm(samples, desc="Processing Samples"):
            run(sample, args)
    else:
        arg_parser.print_help()