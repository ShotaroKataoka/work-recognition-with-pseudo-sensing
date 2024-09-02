import os

import matplotlib.pyplot as plt
import numpy as np
import pywt
from tqdm import tqdm

def get_wavelets_array(frames, start_frame, fps, output_dir, save_wavelets=False):
    """
    Get the wavelets array from the frames.

    Args:
    frames: ndarray, video frames
    start_frame: int, frame number to start sensing from
    fps: int, frames per second of the video
    output_dir: str, path to save the output
    save_wavelets: bool, save the wavelets

    Returns:
    wavelets_array: ndarray, array containing the wavelets coefficients
    wavelets_freqs: ndarray, array containing the frequencies corresponding to the scales
    """
    wavelets_array = []
    rows, cols = frames.shape[1:]
    for i in tqdm(range(rows)):
        for j in range(cols):
            coefficients, freqs = _wavelet_transform(
                frames[:, i, j],
                sampling_fps=fps, 
                start_frame=start_frame,
                cut_duration=10,
                output_file=os.path.join(output_dir, "wavelets", f"wavelet_{i}_{j}.png") if save_wavelets else None
            )
            wavelets_array.append(coefficients)
    wavelets_array = np.array(wavelets_array)
    wavelets_freqs = np.append(freqs, 0.0)
    return wavelets_array, wavelets_freqs

def _wavelet_transform(signal, sampling_fps, start_frame, cut_duration, output_file=None):
    """
    Perform Continuous Wavelet Transform (CWT) and plot the scaleogram.
    
    Args:
    signal: 1D np.array, the signal to analyze
    sampling_fps: int, the sampling frequency of the signal
    start_frame: int, frame number to start sensing from
    cut_duration: int, duration to cut from the start and end of the signal
    output_file: str, path to save the scaleogram plot
    
    Returns:
    coefficients: 2D np.array, the CWT coefficients
    freqs: 1D np.array, the frequencies corresponding to the scales
    """

    wavelet = 'morl'
    dt = 1 / sampling_fps

    scale = sampling_fps / 30
    min_scale = 1 * scale
    max_scale = 150 * scale
    ds = 0.5 * scale
    
    scales = np.arange(min_scale, max_scale+ds, ds)
    scales = np.geomspace(min_scale, max_scale, num=len(scales))
    
    coefficients, freqs = pywt.cwt(signal, scales, wavelet, dt)

    # cut boundary effects
    cut_frames = int(cut_duration * sampling_fps)
    coefficients = coefficients[:, cut_frames:-cut_frames]

    start_time = (start_frame + cut_frames) / sampling_fps
    end_time = (start_frame + len(signal) - cut_frames) / sampling_fps

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(coefficients[::-1, :]), extent=[start_time, end_time, 0, len(freqs)], 
                aspect='auto', vmax=150, vmin=0, cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.ylabel('Frequency [Hz]')
        yticks = np.linspace(0, len(freqs) - 1, num=10)
        ytick_labels = np.round(np.interp(yticks, np.arange(len(freqs)), freqs), 2)
        plt.yticks(yticks, ytick_labels)
        plt.xlabel('Time [sec]')
        plt.title('Wavelet Transform (Scaleogram)')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    return coefficients, freqs

def get_max_freq_index(coefficients, threshold=30):
    """
    Get the index of the frequency component with the highest magnitude.
    
    Args:
    coefficients: 2D np.array, the CWT coefficients
    threshold: int, the threshold to filter out noise
    
    Returns:
    max_freq_index: list, list of the indices of the maximum frequency components
    """
    coefficients = np.where(np.abs(coefficients) < threshold, 0, coefficients)
    max_freq_index = []
    for col in range(coefficients.shape[1]):
        if np.all(coefficients[:, col] == 0):
            max_freq_index.append(-1)
        else:
            max_freq_index.append(np.argmax(np.abs(coefficients[:, col])))

    return max_freq_index
