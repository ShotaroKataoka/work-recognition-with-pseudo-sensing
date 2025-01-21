import json
import os
import time

import numpy as np

from modules.utils import load_video_frames
from modules.wavelet.waveformalizer import waveformalizer
from modules.wavelet.wavelet import get_max_freq_index, get_wavelets_array
from modules.clustering.clustering import clustering_wavelets
from modules.clustering.feature_extractor import pca_wavelets
from modules.classifing_clusters.mean_std_method import classify_clusters
from modules.classifing_clusters.utils import load_sensor, save_cluster_class, copy_wavelet_to_each_cluster_dir, save_sensor


def determine_sensing_points(
        video_path, 
        experiment_name,
        output_name, 
        fps, 
        start_frame, 
        duration_sec, 
        filter_size=(4, 4),
        saves={}
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
    saves: dict, dictionary of the intermediate saves
    """

    save_all = saves.get('save_all', False)
    save_grayscale = saves.get('save_grayscale', False)
    save_filtered = saves.get('save_filtered', False)
    save_waveforms = saves.get('save_waveforms', False)
    save_wavelets = saves.get('save_wavelets', False)
    save_cluster_grid = saves.get('save_cluster_grid', False)
    save_cluster_scatter = saves.get('save_cluster_scatter', False)
    
    output_dir = os.path.join("pseudo_sensing_output", experiment_name, output_name)
    duration_frames = int(np.ceil(duration_sec * fps))
    processing_time = {}

    print("1. Performing Wavelet Transform")
    
    processing_time['1. load video frames'] = time.time()
    frames = load_video_frames(video_path, start_frame, duration_frames)
    processing_time['1. load video frames'] = time.time() - processing_time['1. load video frames']
    processing_time['1. get_wavelets_array'] = time.time()
    frames = waveformalizer(frames, output_dir, fps, start_frame, filter_size=filter_size, save_grayscale=save_grayscale or save_all, save_filtered=save_filtered or save_all, save_waveforms=save_waveforms or save_all)
    wavelets_array, wavelet_freqs = get_wavelets_array(frames, start_frame, fps, output_dir, save_wavelets=save_wavelets or save_all)
    max_freq_indicess = [get_max_freq_index(wavelets_array[i]) for i in range(wavelets_array.shape[0])]
    mean_max_magnitudes = []
    for i in range(wavelets_array.shape[0]):
        mean_max_magnitudes += [np.mean(np.abs(wavelets_array[i][max_freq_indicess[i]]))]
    processing_time['1. get_wavelets_array'] = time.time() - processing_time['1. get_wavelets_array']
    
    print("2. Performing Clustering")
    processing_time['2. pca'] = time.time()
    wavelets_feat_array = pca_wavelets(wavelets_array, n_components=50)
    processing_time['2. pca'] = time.time() - processing_time['2. pca']
    processing_time['2. clustering'] = time.time()
    wavelets_cluster_labels = clustering_wavelets(wavelets_feat_array, output_dir, frames.shape[1], frames.shape[2], save_cluster_grid=save_cluster_grid or save_all, save_cluster_scatter=save_cluster_scatter or save_all)
    processing_time['2. clustering'] = time.time() - processing_time['2. clustering']

    print("3. Performing Classifing Clusters")
    processing_time['3. classify_clusters'] = time.time()
    cluster_class = classify_clusters(max_freq_indicess, wavelet_freqs, wavelets_cluster_labels)
    processing_time['3. classify_clusters'] = time.time() - processing_time['3. classify_clusters']
    save_cluster_class(wavelets_cluster_labels, cluster_class, output_dir, frames.shape[1], frames.shape[2], save_fig=True)
    copy_wavelet_to_each_cluster_dir(output_dir, wavelets_cluster_labels, cluster_class, frames.shape[1], frames.shape[2])

    save_sensor(output_dir, wavelets_cluster_labels, cluster_class, filter_size, fps, frames.shape[1], frames.shape[2], mean_max_magnitudes)
    save_experiment_info(output_dir, fps, start_frame, filter_size, duration_sec, video_path, frames.shape[1:], processing_time, saves)

def save_experiment_info(output_dir, fps, start_frame, filter_size, duration_sec, video_path, grid_size, processing_time, saves):
    """
    Save the experiment settings as a JSON file.
    
    Args:
    output_dir: str, path to the output directory
    fps: int, frames per second of the video
    start_frame: int, frame number to start sensing from
    filter_size: tuple, size of the filter to apply (width, height)
    duration_sec: int, duration of the sensing in seconds
    video_path: str, path to the video file
    grid_size: tuple, number of rows and columns in the grid
    processing_time: dict, dictionary of processing times
    saves: dict, dictionary of the intermediate saves
    """

    waveformalizer = {'filter_size': filter_size, 'duration_sec': duration_sec, 'method': 'grayscale-mean', 'grid_size': grid_size}
    scales = """    dt = 1 / sampling_fps
    scale = sampling_fps / 30
    min_scale = 1 * scale
    max_scale = 150 * scale
    ds = 0.5 * scale
    scales = np.arange(min_scale, max_scale+ds, ds)
    scales = np.geomspace(min_scale, max_scale, num=len(scales))"""
    wavelet = {'wavelet': 'morl', 'sampling_fps': fps, 'scales': scales, 'cut_frames': int(fps * 10), 'threshold': 30}
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
        'classification_method': classification_method,
        'processing_time': processing_time,
        'saves': saves
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'exp_settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)

def run_sensing(video_path, sensor_dir, start_frame=0, duration_frames=None, top_k=3, cut_duration=10, median_window_size=5, output_dir=None):
    """
    Run the sensing for the video.
    
    Args:
    frames: ndarray, frames of the video
    sensor_dir: str, path to the sensor directory
    start_frame: int, frame number to start sensing from, default 0
    duration_frames: int, duration of the sensing in frames, default None
    top_k: int, number of top sensing grids to select, default 3
    cut_duration: int, duration of the cut in seconds, default 10
    median_window_size: int, size of the window for median filter, default 3
    output_dir: str, path to the output directory, default None
    """
    sensor = load_sensor(sensor_dir)
    fps = sensor['fps']
    filter_size = sensor['filter_size']
    grids = sensor['grids']
    
    frames = load_video_frames(video_path, start_frame, duration_frames)
    frames = reflect_padding(frames, cut_duration * fps)

    sensing_grids_indices = get_sensing_grid_index(grids, top_k=top_k)
    frames = waveformalizer(frames, None, fps, 0, filter_size=filter_size)
    wavelets_array, wavelet_freqs = get_wavelets_array(frames, 0, fps, output_dir, 
                                                       save_wavelets=True if output_dir else False,
                                                       exec_index=sensing_grids_indices, 
                                                       cut_duration=cut_duration)
    max_freq_indicess = [get_max_freq_index(wavelets_array[i], threshold=30) for i in range(wavelets_array.shape[0])]
 
    # coeffs = []
    # for i, max_freq_indices in enumerate(max_freq_indicess):
    #     coeff = []
    #     for t, max_freq_index in enumerate(max_freq_indices):
    #         coeff.append(wavelets_array[i, max_freq_index, t])
    #     coeffs.append(coeff)
    # coeffs = np.array(coeffs)
    # max_coeffs_sensors_index = np.argmax(np.abs(coeffs), axis=0)
    # speeds = []
    # for t, max_coeffs_sensor_index in enumerate(max_coeffs_sensors_index):
    #     max_freq_index = max_freq_indicess[max_coeffs_sensor_index][t]
    #     speed = wavelet_freqs[max_freq_index]
    #     speeds.append(speed)
    # speeds = np.array(speeds)

    speeds = []
    for i in range(len(max_freq_indicess[0])):
        speed = []
        for max_freq_indices in max_freq_indicess:
            speed.append(wavelet_freqs[max_freq_indices[i]])
        median_value = np.median(speed)
        speeds.append(median_value)

    filtered_speeds = []
    for i in range(len(speeds)):
        window_start = max(0, i - median_window_size // 2)
        window_end = min(len(speeds), i + median_window_size // 2 + 1)
        window = speeds[window_start:window_end]
        median_value = np.median(window)
        filtered_speeds.append(median_value)
    return filtered_speeds

def reflect_padding(frames, pad_size):
    """
    Apply reflect padding to the signal.
    
    Args:
    frames: ndarray, frames of the video
    pad_size: int, size of the padding
    
    Returns:
    ndarray: The padded signal
    """
    pad_size = int(pad_size)
    return np.concatenate([frames[:pad_size][::-1], frames, frames[-pad_size:][::-1]])

def get_sensing_grid_index(grids, top_k=None):
    """
    Select the top k sensing grids based on the mean max magnitude.

    Args:
    grids: list, list of sensing grids
    top_k: int, number of top sensing grids to select

    Returns:
    ndarray: The selected sensing grid
    """
    detection_grids = grids['Detection Grid']
    detection_grids = sorted(detection_grids, key=lambda x: x['mean_max_magnitude'], reverse=True)

    all_grids = grids["Stationary Grid"] + detection_grids + grids["Noise Grid"]
    num_cols = max([grid['col'] for grid in all_grids]) + 1

    indices = []
    for detection_grid in detection_grids:
        row = detection_grid['row']
        col = detection_grid['col']
        indices.append(row * num_cols + col)
    
    if top_k is not None:
        indices = indices[:top_k]
    
    return indices
        