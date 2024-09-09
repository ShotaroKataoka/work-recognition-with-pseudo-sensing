import json
import os

import numpy as np

from modules.utils import load_video_frames
from modules.wavelet.waveformalizer import waveformalizer
from modules.wavelet.wavelet import get_wavelets_array
from modules.clustering.clustering import clustering_wavelets
from modules.clustering.feature_extractor import pca_wavelets
from modules.classifing_clusters.mean_std_method import classify_clusters
from modules.classifing_clusters.utils import save_cluster_class, copy_wavelet_to_each_cluster_dir, save_sensors


def determine_sensing_points(
        video_path, 
        experiment_name,
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
        save_cluster_scatter=False
    ):
    """
    Determine sensing points for the video.
    
    Args:
    video_path: str, path to the video file
    experiment_name: str, name of the experiment
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
    save_cluster_scatter: bool, save the cluster scatter plot
    """

    output_dir = os.path.join("pseudo_sensing_output", experiment_name, output_name)
    save_experiment_settings(fps, start_frame, filter_size, duration_sec, video_path, output_dir)

    duration_frames = int(np.ceil(duration_sec * fps))

    print("1. Performing Wavelet Transform")
    frames = load_video_frames(video_path, start_frame, duration_frames)
    frames = waveformalizer(frames, output_dir, fps, start_frame, filter_size=filter_size, save_grayscale=save_grayscale or save_all, save_filtered=save_filtered or save_all, save_waveforms=save_waveforms or save_all)
    
    wavelets_array, wavelet_freqs = get_wavelets_array(frames, start_frame, fps, output_dir, save_wavelets=save_wavelets or save_all)
    
    print("2. Performing Clustering")
    wavelets_feat_array = pca_wavelets(wavelets_array, n_components=50)
    wavelets_cluster_labels = clustering_wavelets(wavelets_feat_array, output_dir, frames.shape[1], frames.shape[2], save_cluster_grid=save_cluster_grid or save_all, save_cluster_scatter=save_cluster_scatter or save_all)

    print("3. Performing Classifing Clusters")
    cluster_class = classify_clusters(wavelets_array, wavelet_freqs, wavelets_cluster_labels)
    save_cluster_class(wavelets_cluster_labels, cluster_class, output_dir, frames.shape[1], frames.shape[2], save_fig=True)
    copy_wavelet_to_each_cluster_dir(output_dir, wavelets_cluster_labels, cluster_class, frames.shape[1], frames.shape[2])

    save_sensors(output_dir, wavelets_cluster_labels, cluster_class, filter_size, rows=frames.shape[1], cols=frames.shape[2])

def save_experiment_settings(fps, start_frame, filter_size, duration_sec, video_path, output_dir):
    """
    Save the experiment settings as a JSON file.
    
    Args:
    fps: int, frames per second of the video
    start_frame: int, frame number to start sensing from
    filter_size: tuple, size of the filter to apply (width, height)
    duration_sec: int, duration of the sensing in seconds
    video_path: str, path to the video file
    output_dir: str, path to the output directory
    """

    waveformalizer = {'filter_size': filter_size, 'duration_sec': duration_sec, 'method': 'grayscale-mean'}
    scales = """    dt = 1 / sampling_fps
    scale = sampling_fps / 30
    min_scale = 1 * scale
    max_scale = 150 * scale
    ds = 0.5 * scale
    scales = np.arange(min_scale, max_scale+ds, ds)
    scales = np.geomspace(min_scale, max_scale, num=len(scales))"""
    wavelet = {'wavelet': 'morl', 'sampling_fps': fps, 'scales': scales, 'cut_frames': int(fps * 10)}
    wavelet_method = {'waveformalizer': waveformalizer, 'wavelet': wavelet}
    
    feature_extractor = {'method': 'pca', 'params': {'n_components': 50}, 'preprocessing': """    epsilon=1e-6
    mean = np.mean(wavelets_array, axis=0)
    std = np.std(wavelets_array, axis=0) + epsilon
    wavelets_array = (wavelets_array - mean) / std"""}
    clustering = {'method': 'kmeans', 'params': {'n_clusters': 3, 'random_state': 0, 'n_init': 'auto'}}
    clustering_method = {'feature_extractor': feature_extractor, 'clustering': clustering}

    classification_method = {'method': 'mean_std', 'description': """Stationary Grid: The cluster with the minimum mean frequency is the stationary grid.
Detection Grid: The cluster with the maximum standard deviation of the frequency indices is the detection grid.
Noise Grid: The remaining cluster is the noise grid."""}
    settings = {
        'fps': fps,
        'start_frame': start_frame,
        'video_path': video_path,
        'wavelet_method': wavelet_method,
        'clustering_method': clustering_method,
        'classification_method': classification_method
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'exp_settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)

# def run_sensing(video_path, output_name, sensing_