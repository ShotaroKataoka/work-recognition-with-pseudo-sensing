import os

import cv2
from tqdm import tqdm


def crop_regions(video_path, regions, output_files):
    """
    Save cropped regions of the video to separate files.

    Args:
    video_path: str, path to the input video file
    regions: list of tuples (x, y, w, h), where (x, y) is the top-left corner of the region,
             and (w, h) are the width and height of the cropped area
    output_files: list of str, corresponding output file names for each region
    """

    if len(regions) != len(output_files):
        raise ValueError("Number of regions and output files should match")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Could not open video file")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    writers = []
    for i, region in enumerate(regions):
        x, y, w, h = region
        cropped_frame_size = (w, h)
        output_file = output_files[i]
        os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
        writer = cv2.VideoWriter(output_file, fourcc, fps, cropped_frame_size)
        writers.append(writer)

    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            for i, region in enumerate(regions):
                x, y, w, h = region
                cropped_frame = frame[y:y+h, x:x+w]
                writers[i].write(cropped_frame)

            pbar.update(1)

    for writer in writers:
        writer.release()

    video.release()
