import os

import numpy as np

from modules.utils import load_video_frames, save_frames_as_video, save_waveform_plot
from modules.wavelet.waveformalizer import to_grayscale, mean_filter
from modules.wavelet.wavelet import get_wavelets_array, get_max_freq_index
from modules.clustering.clustering import clustering_wavelets, save_cluster_grid_image, visualize_clustering
from modules.clustering.feature_extractor import pca_wavelets, tsne_wavelets

def determine_sensing_points(
        video_path, 
        output_name, 
        fps, 
        start_frame, 
        duration_sec, 
        filter_size=(4, 4),
        save_all=False,
        save_grayscale=False,
        save_filtered=False,
        save_waveforms=False,
        save_wavelets=False,
        save_cluster_grid=False
    ):
    """
    Determine sensing points for the video.
    
    Args:
    video_path: str, path to the video file
    output_name: str, name of the output
    fps: int, frames per second of the video
    start_frame: int, frame number to start sensing from
    duration_sec: int, duration of the sensing in seconds
    filter_size: tuple, size of the filter to apply (width, height), default (4, 4)
    save_all: bool, save all the intermediate
    save_grayscale: bool, save the grayscale video
    save_filtered: bool, save the filtered video
    save_waveforms: bool, save the waveforms
    save_wavelets: bool, save the wavelets
    save_cluster_grid: bool, save the cluster grid

    Returns:
    sensing_points: list, list of dictionaries containing the sensing points
    """
    
    output_dir = os.path.join("pseudo_sensing_output", output_name)

    duration_frames = int(np.ceil(duration_sec * fps))
    frames = load_video_frames(video_path, start_frame, duration_frames)
    frames = to_grayscale(frames)
    if save_grayscale or save_all:
        save_frames_as_video(frames, os.path.join(output_dir, "grayscale.mp4"), fps)
    frames = mean_filter(frames, filter_size)
    if save_filtered or save_all:
        save_frames_as_video(frames, os.path.join(output_dir, "filtered.mp4"), fps)
    if save_waveforms or save_all:
        save_waveform_plot(frames, output_dir, start_frame, fps)
    
    wavelets_array, wavelets_freqs = get_wavelets_array(frames, start_frame, fps, output_dir, save_wavelets=save_wavelets, save_all=save_all)
    
    print("Performing feat extraction and Clustering on Wavelets")
    wavelets_feat_array = pca_wavelets(wavelets_array, n_components=50)
    wavelets_cluster_labels = clustering_wavelets(wavelets_feat_array)

    if save_cluster_grid or save_all:
        save_cluster_grid_image(output_dir, labels=wavelets_cluster_labels, rows=frames.shape[1], cols=frames.shape[2])
    
    print("Determining Sensing Points")
    wavelets_means, wavelets_stds = calculate_stationarity_score(wavelets_array, wavelets_freqs)
    
    print("Visualizing t-SNE and Clustering Results")
    additional_info = [{"Mean": mean, "Std": std} for mean, std in zip(wavelets_means, wavelets_stds)]
    visualize_clustering(wavelets_array, wavelets_cluster_labels, additional_info=additional_info)
    exit()

    
    sensing_points = []
    return sensing_points


def calculate_stationarity_score(wavelets_array, wavelet_freqs):
    """
    Calculate the stationarity score for each pixel.
    
    Args:
    wavelets_array: ndarray, array containing the wavelets coefficients
    
    Returns:
    means: ndarray, array containing the means
    stds: ndarray, array containing the standard deviations
    """
    means = []
    stds = []
    for i in range(wavelets_array.shape[0]):
        max_freq_indices = get_max_freq_index(wavelets_array[i])
        max_freqs = wavelet_freqs[max_freq_indices]
        means.append(np.mean(max_freqs))
        stds.append(np.std(max_freqs))
    means = np.array(means)
    stds = np.array(stds)
    return means, stds