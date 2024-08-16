from glob import glob
import os

import cv2

from pseudo_sensing import pseudo_sensing
from video_processing import load_video

def run_sample():
    video_files = glob("sample_video/*.mp4")
    if not video_files:
        print("No video files found.")
        return
    
    for video_file in video_files:
        print(f"Processing {video_file}...")
        video = load_video(video_file)
        pseudo_sensing(video)
        print("Processing completed.")

if __name__ == "__main__":
    run_sample()
    print("All done.")