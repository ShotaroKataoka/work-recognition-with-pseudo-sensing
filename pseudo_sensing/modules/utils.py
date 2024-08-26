import json
import os

import cv2
import numpy as np

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

def load_video_frames(video_path, start_frame, duration_frames, rgb=True):
    """Load video frames as ndarray.
    
    Args:
        video_path (str): Path to the video file.
        start_frame (int): Frame number to start sensing from.
        duration_frames (int): Number of frames to sense.
        rgb (bool): Whether to load the frames in RGB format. Default is True.
    
    Returns:
        ndarray: A 4D numpy array containing the video frames.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for i in range(duration_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
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
