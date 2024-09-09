import json
import os

import numpy as np
from tqdm import tqdm

from modules.utils import load_samples_json, load_video_frames, save_frames_as_video


def make_test_video(sample):
    """
    Make test data from a sample.
    
    Args:
    sample: dict, sample data
    save_test_json: bool, whether to save the test data as a json file. Default is False.

    Returns:
    dict: A dictionary containing the test data.
    """
    video_dir = f"sample_video/cropped"
    fps = sample['info']['frame_rate']
    start_secs = [30, 600, 1200]
    DURATION_SECONDS = 30
    test_data = {}
    for i, sensor in enumerate(sample['sensors']):
        video_path = os.path.join(video_dir, f"{sample['id']}/{i}.mp4")
        for start_sec in start_secs:
            start_frame = start_sec * fps
            duration_frames = int(np.ceil(DURATION_SECONDS * fps))
            output_name = f"{sample['id']}_{i}_{sensor['movement']['type']}_{sensor['movement']['direction']}_{start_sec}"
            frames = load_video_frames(video_path, start_frame, duration_frames)
            save_frames_as_video(frames, f"sample_video/test_data/{output_name}.mp4", fps)
            test_data[output_name] = {
                'id': sample['id'],
                'sensor_id': i,
                'movement_type': sensor['movement']['type'],
                'movement_direction': sensor['movement']['direction'],
                'start_frame': start_frame,
                'duration_frames': duration_frames,
                'fps': fps,
                'videopath': f"sample_video/test_data/{output_name}.mp4",
                'count': 0
            }
    return test_data

if __name__ == "__main__":
    samples = load_samples_json()
    test_data = {}
    for sample in tqdm(samples, desc="Processing Samples"):
        test_data_ = make_test_video(sample)
        test_data.update(test_data_)

    save_test_json = True
    if save_test_json:
        with open("sample_video/test_data/test.json", "w") as f:
            json.dump(test_data, f, indent=4)