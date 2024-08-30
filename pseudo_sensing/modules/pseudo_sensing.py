import os

import numpy as np
import matplotlib.pyplot as plt
import pywt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from modules.utils import load_video_frames, save_frames_as_video

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
        save_plot_waveform(frames, output_dir, start_frame, fps)
    
    rows, cols = frames.shape[1], frames.shape[2]
    wavelets_array = []
    for i in tqdm(range(rows)):
        for j in range(cols):
            coefficients, freqs = wavelet_transform(
                frames[:, i, j],
                sampling_fps=fps, 
                start_frame=start_frame,
                cut_duration=10,
                output_file=os.path.join(output_dir, "wavelets", f"wavelet_{i}_{j}.png") if save_wavelets or save_all else None
            )
            wavelets_array.append(coefficients)
    wavelets_array = np.array(wavelets_array)
    wavelets_freqs = np.append(freqs, 0.0)
    
    print("Performing feat extraction and Clustering on Wavelets")
    wavelets_feat_array = pca_wavelets(wavelets_array, n_components=50)
    wavelets_cluster_labels = clustering_wavelets(wavelets_feat_array)

    if save_cluster_grid or save_all:
        save_cluster_grid_image(output_dir, labels=wavelets_cluster_labels, rows=rows, cols=cols)
    
    print("Determining Sensing Points")
    wavelets_means, wavelets_stds = calculate_stationarity_score(wavelets_array, wavelets_freqs)
    
    print("Visualizing t-SNE and Clustering Results")
    import mplcursors
    wavelets_tsne_array = tsne_wavelets(wavelets_array, n_components=2)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(wavelets_tsne_array[:, 0], wavelets_tsne_array[:, 1], c=wavelets_cluster_labels, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE of Wavelets with Clustering')
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(f'Index: {sel.target.index} \nMean: {wavelets_means[sel.target.index]:.2f} \nStd: {wavelets_stds[sel.target.index]:.2f}')

    plt.show()
    exit()

    
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

def wavelet_transform(signal, sampling_fps, start_frame, cut_duration, output_file=None):
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

def pca_wavelets(wavelets_array, n_components=100):
    """
    Perform PCA on the wavelets grid.
    
    Args:
    wavelets_array: ndarray, array containing the wavelets coefficients
    
    Returns:
    wavelets_pca_grid: dict, dictionary containing the PCA components
    """
    wavelets_array = wavelets_array.reshape(wavelets_array.shape[0], -1)
    wavelets_array = wavelets_array - np.mean(wavelets_array, axis=0) / np.std(wavelets_array, axis=0)

    pca = PCA(n_components=n_components, random_state=0)
    wavelets_pca_array = pca.fit_transform(wavelets_array)
    return wavelets_pca_array

def tsne_wavelets(wavelets_array, n_components=3):
    """
    Perform t-SNE on the wavelets grid.
    
    Args:
    wavelets_array: ndarray, array containing the wavelets coefficients

    Returns:
    wavelets_tsne_array: ndarray, array containing the t-SNE components
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=n_components, random_state=0)
    wavelets_array = wavelets_array.reshape(wavelets_array.shape[0], -1)
    wavelets_tsne_array = tsne.fit_transform(wavelets_array)
    return wavelets_tsne_array

def clustering_wavelets(wavelets_pca_array):
    """
    Perform clustering on the wavelets grid.
    
    Args:
    wavelets_pca_array: ndarray, array containing the PCA components
    
    Returns:
    cluster_labels: ndarray, array containing the cluster labels
    """

    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(wavelets_pca_array)
    return cluster_labels

def save_cluster_grid_image(output_dir, labels, rows, cols):
    """
    Save the cluster grid as an image.
    
    Args:
    output_dir: str, path to save the image
    labels: ndarray, array containing the cluster labels
    rows: int, number of rows in the grid
    cols: int, number of columns in the grid
    """

    cluster_grid = labels.reshape(rows, cols)
    plt.figure(figsize=(8, 8))
    plt.imshow(cluster_grid, cmap='viridis')
    plt.colorbar()
    plt.title('Cluster Grid')
    plt.savefig(os.path.join(output_dir, 'cluster_grid.png'))
    plt.close()

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