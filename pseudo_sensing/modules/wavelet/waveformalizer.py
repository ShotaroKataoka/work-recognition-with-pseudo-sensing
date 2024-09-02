import os

import numpy as np

from modules.utils import save_frames_as_video, save_waveform_plot

def waveformalizer(
        frames, 
        output_dir, 
        fps, 
        start_frame, 
        filter_size=(4, 4),
        save_grayscale=False,
        save_filtered=False,
        save_waveforms=False
    ):
    """
    Perform waveformalization on the frames.

    Args:
    frames: ndarray, video frames
    output_dir: str, path to save the output
    fps: int, frames per second of the video
    start_frame: int, frame number to start sensing from
    filter_size: tuple, size of the filter to apply (width, height), default (4, 4)
    save_grayscale: bool, save the grayscale video
    save_filtered: bool, save the filtered video
    save_waveforms: bool, save the waveforms

    Returns:
    frames: ndarray, waveformalized video frames
    """
    frames = _to_grayscale(frames)
    if save_grayscale:
        save_frames_as_video(frames, os.path.join(output_dir, "grayscale.mp4"), fps)
    frames = _mean_filter(frames, filter_size)
    if save_filtered:
        save_frames_as_video(frames, os.path.join(output_dir, "filtered.mp4"), fps)
    if save_waveforms:
        save_waveform_plot(frames, output_dir, start_frame, fps)
    return frames

def _to_grayscale(frames):
    """
    Convert frames to grayscale.
    
    Args:
    frames: ndarray, video frames

    Returns:
    grayscale_frames: ndarray, grayscale video frames
    """
    grayscale_frames = np.mean(frames, axis=-1)
    return grayscale_frames

def _mean_filter(frames, filter_size):
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
