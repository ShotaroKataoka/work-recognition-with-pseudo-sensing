import os
import argparse

from tqdm import tqdm

from modules.pseudo_sensing import run_sensing
from modules.utils import load_samples_json

def run():
    """Run pseudo sensing on a sample.

    Args:
    sample: dict, sample data
    args: argparse.Namespace, arguments
    """
    
    sensor = 'mIFuCipTWkQ_2_Conveyor Motion_Left'
    # sensor = 'L6rzLUJ1QoA_0_Conveyor Motion_Forward'
    video = 'mIFuCipTWkQ/2.mp4'
    # video = 'L6rzLUJ1QoA/0.mp4'

    sensor_dir = os.path.join("pseudo_sensing_output", 'exp01', sensor)
    video_path = os.path.join("sample_video/cropped", video)

    run_sensing(video_path, sensor_dir)


if __name__ == "__main__":
    run()