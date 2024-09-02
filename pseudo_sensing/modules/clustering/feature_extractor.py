from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def pca_wavelets(wavelets_array, n_components=100):
    """
    Perform PCA on the wavelets grid.
    
    Args:
    wavelets_array: ndarray, array containing the wavelets coefficients
    n_components: int, number of components to keep
    
    Returns:
    wavelets_pca_grid: dict, dictionary containing the PCA components
    """
    wavelets_array = wavelets_array.reshape(wavelets_array.shape[0], -1)
    
    epsilon=1e-6
    mean = np.mean(wavelets_array, axis=0)
    std = np.std(wavelets_array, axis=0) + epsilon
    wavelets_array = (wavelets_array - mean) / std

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

    tsne = TSNE(n_components=n_components, random_state=0)
    wavelets_array = wavelets_array.reshape(wavelets_array.shape[0], -1)
    wavelets_tsne_array = tsne.fit_transform(wavelets_array)
    return wavelets_tsne_array