import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_samples_json():
    """Load sample.json and sample_info.json and merge them.
    
    Returns:
        list: A list of dictionaries containing the merged data.
    
    --- merged samples ---
    [{
        'id': string,
        'sensers': [{
            'x': int, 
            'y': int, 
            'w': int, 
            'h': int, 
            'start_frame': int, 
            'movement': {
                'type': str, 
                'direction': str,
            }
        }, ...],
        'filename': str, 
        'info': {
            'frame_rate': int,
            'nb_frames': int,
            'width': int,
            'height': int,
            'duration': int
        }
    }, ...]
    """
    with open('sample_video/sample.json', 'r') as f:
        data = json.load(f)

    with open('sample_video/sample_info.json', 'r') as f:
        info = json.load(f)

    merged = []

    for d in data:
        for i in info:
            if d['id'] == i['id']:
                d.update(i)
                merged.append(d)
    return merged

def load_video_frames(video_path, start_frame, duration_frames=None, rgb=True):
    """Load video frames as ndarray.
    
    Args:
        video_path (str): Path to the video file.
        start_frame (int): Frame number to start sensing from.
        duration_frames (int or None): Number of frames to sense, or None to read until the end.
        rgb (bool): Whether to load the frames in RGB format. Default is True.
    
    Returns:
        ndarray: A 4D numpy array containing the video frames.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    read_frames = 0
    while duration_frames is None or read_frames < duration_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        read_frames += 1
    
    cap.release()
    
    return np.array(frames)

def save_frames_as_video(frames, output_path, fps=30):
    """Save ndarray frames as a video file.
    
    Args:
        frames (ndarray): color or grayscale video frames. tensor shape (n_frames, height, width, 3) or (n_frames, height, width).
        output_path (str): Path to save the video file.
        fps (int): Frames per second of the video. Default is 30.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_height, frame_width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # Ensure frames are 8-bit unsigned
    frames = frames.astype(np.uint8)
    
    # If grayscale, convert to 3-channel color
    if frames.ndim == 3:
        frames = np.stack((frames,) * 3, axis=-1)
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def save_waveform_plot(frames, output_dir, start_frame=0, fps=30):
    """
    Plot all of the temporal change of each pixel in a video
    y-axis: pixel value (0-255)
    
    Args:
    frames: ndarray, video frames
    output_dir: str, path to save the plots
    start_frame: int, frame number to start sensing from
    fps: int, frames per second of the video
    """
    
    n_frames, frame_height, frame_width = frames.shape

    for i in range(frame_height):
        for j in range(frame_width):
            filename = os.path.join(output_dir, "waveforms", f"pixel_{i}_{j}.png")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            time_axis = np.arange(start_frame, start_frame + n_frames) / fps
            plt.plot(time_axis, frames[:, i, j])
            plt.xlabel("Time (s)")
            plt.ylabel("Pixel Value")
            plt.ylim(0, 255)
            plt.title(f"Temporal Change of Pixel ({i}, {j})")
            xticks = np.arange(start_frame // fps, (start_frame + n_frames) // fps + 3, 5)
            plt.xticks(xticks)
            plt.savefig(filename)
            plt.close()