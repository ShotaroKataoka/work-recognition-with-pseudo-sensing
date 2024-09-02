import numpy as np

from modules.wavelet.wavelet import get_wavelets_array, get_max_freq_index

def classify_clusters(wavelets_array, wavelet_freqs, wavelets_cluster_labels):
    """
    Classify the clusters using the mean and standard deviation method.

    Args:
    wavelets_array: ndarray, array containing the wavelets coefficients
    wavelet_freqs: ndarray, array containing the frequencies corresponding to the scales
    wavelets_cluster_labels: ndarray, array containing the cluster labels

    Returns:
    cluster_class: dict, dictionary containing the cluster classes
    """
    wavelets_means, wavelets_stds = calculate_stationarity_score(wavelets_array, wavelet_freqs)
    mean_means, mean_stds, count = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}
    for mean, std, cluster in zip(wavelets_means, wavelets_stds, wavelets_cluster_labels):
        mean_means[cluster] += mean
        mean_stds[cluster] += std
        count[cluster] += 1
    for c in range(3):
        mean_means[c] /= count[c]
        mean_stds[c] /= count[c]

    cluster_class = {}
    
    minimum_mean_cluster = min(mean_means, key=mean_means.get)
    cluster_class[minimum_mean_cluster] = 'Stationary Grid'

    remaining_clusters = list(set([0, 1, 2]) - set([minimum_mean_cluster]))
    if mean_stds[remaining_clusters[0]] < mean_stds[remaining_clusters[1]]:
        cluster_class[remaining_clusters[0]] = 'Detection Grid'
        cluster_class[remaining_clusters[1]] = 'Noise Grid'
    else:
        cluster_class[remaining_clusters[1]] = 'Detection Grid'
        cluster_class[remaining_clusters[0]] = 'Noise Grid'
    
    return cluster_class

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