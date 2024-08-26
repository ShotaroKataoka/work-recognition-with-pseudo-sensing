import os

import numpy as np
import matplotlib.pyplot as plt

from modules.utils import load_video_frames, save_frames_as_video

def determine_sensing_points(video_path, output_name, fps, start_frame, duration_sec, filter_size=(4, 4), vobose_visualization=False):
    """
    Determine sensing points for the video.
    
    Args:
    video_path: str, path to the video file
    output_name: str, name of the output
    fps: int, frames per second of the video
    start_frame: int, frame number to start sensing from
    duration_sec: int, duration of the sensing in seconds
    filter_size: tuple, size of the filter to apply (width, height), default (4, 4)
    vobose_visualization: bool, whether to save verbose visualization, default False

    Returns:
    sensing_points: list, list of dictionaries containing the sensing points
    """
    
    output_dir = os.path.join("pseudo_sensing_output", output_name)

    duration_frames = int(duration_sec * fps)
    frames = load_video_frames(video_path, start_frame, duration_frames)
    frames = to_grayscale(frames)
    if vobose_visualization:
        save_frames_as_video(frames, os.path.join(output_dir, "grayscale.mp4"), fps)
    frames = mean_filter(frames, filter_size)
    if vobose_visualization:
        save_frames_as_video(frames, os.path.join(output_dir, "filtered.mp4"), fps)
        save_plot_waveform(frames, output_dir, start_frame, fps)
    
    sensing_points = []
    return sensing_points

def to_grayscale(frames):
    """
    Convert frames to grayscale.
    
    Args:
    frames: ndarray, video frames

    Returns:
    grayscale_frames: ndarray, grayscale video frames
    """
    grayscale_frames = np.mean(frames, axis=-1)
    return grayscale_frames

def mean_filter(frames, filter_size):
    """
    Apply filter to the frames.
    
    Args:
    frames: ndarray, video frames
    filter_size: tuple, size of the filter to apply (width, height)

    Returns:
    filtered_frames: ndarray, filtered video frames
    """
    filter_width, filter_height = filter_size
    n_frames, frame_height, frame_width = frames.shape
    
    # Ensure the frames dimensions are divisible by filter_size
    if frame_height % filter_height != 0 or frame_width % filter_width != 0:
        raise ValueError("Frame dimensions must be divisible by filter_size without remainder")

    # Calculate the number of subregions
    new_height = frame_height // filter_height
    new_width = frame_width // filter_width

    # Reshape and apply filter by averaging over the defined regions
    reshaped_frames = frames.reshape(n_frames, new_height, filter_height, new_width, filter_width)
    mean_filtered_frames = reshaped_frames.mean(axis=(2, 4))
    
    return mean_filtered_frames

def save_plot_waveform(frames, output_dir, start_frame=0, fps=30):
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