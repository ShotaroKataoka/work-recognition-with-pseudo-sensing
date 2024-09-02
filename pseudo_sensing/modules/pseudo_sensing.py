import os

import numpy as np

from modules.utils import load_video_frames
from modules.wavelet.waveformalizer import waveformalizer
from modules.wavelet.wavelet import get_wavelets_array
from modules.clustering.clustering import clustering_wavelets
from modules.clustering.feature_extractor import pca_wavelets
from modules.classifing_clusters.mean_std_method import classify_clusters
from modules.classifing_clusters.utils import save_cluster_class, copy_wavelet_to_each_cluster_dir

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
        save_cluster_grid=False,
        save_cluster_scater=False
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
    save_cluster_scater: bool, save the cluster scatter plot

    Returns:
    sensing_points: list, list of dictionaries containing the sensing points
    """
    
    output_dir = os.path.join("pseudo_sensing_output", output_name)

    duration_frames = int(np.ceil(duration_sec * fps))

    print("1. Performing Wavelet Transform")
    frames = load_video_frames(video_path, start_frame, duration_frames)
    frames = waveformalizer(frames, output_dir, fps, start_frame, filter_size=filter_size, save_grayscale=save_grayscale or save_all, save_filtered=save_filtered or save_all, save_waveforms=save_waveforms or save_all)
    wavelets_array, wavelet_freqs = get_wavelets_array(frames, start_frame, fps, output_dir, save_wavelets=save_wavelets or save_all)

    print("2. Performing Clustering")
    wavelets_feat_array = pca_wavelets(wavelets_array, n_components=50)
    wavelets_cluster_labels = clustering_wavelets(wavelets_feat_array, output_dir, frames.shape[1], frames.shape[2], save_cluster_grid=save_cluster_grid or save_all, save_cluster_scater=save_cluster_scater or save_all)
    
    print("3. Performing Classifing Clusters")
    cluster_class = classify_clusters(wavelets_array, wavelet_freqs, wavelets_cluster_labels)
    save_cluster_class(wavelets_cluster_labels, cluster_class, output_dir, frames.shape[1], frames.shape[2], save_fig=True)
    copy_wavelet_to_each_cluster_dir(output_dir, wavelets_cluster_labels, cluster_class, frames.shape[1], frames.shape[2])
    
    sensing_points = []
    return sensing_points
