import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

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