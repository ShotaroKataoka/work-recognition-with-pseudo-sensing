import os

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from modules.clustering.feature_extractor import tsne_wavelets

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

def visualize_clustering(wavelets_array, wavelets_cluster_labels, additional_info=None):
    """
    Visualize the clustering results.

    Args:
    wavelets_array: ndarray, array containing the wavelets coefficients
    wavelets_cluster_labels: ndarray, array containing the cluster labels
    additional_info: list, list of additional information dict to display
    """
    import mplcursors
    wavelets_tsne_array = tsne_wavelets(wavelets_array, n_components=2)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(wavelets_tsne_array[:, 0], wavelets_tsne_array[:, 1], c=wavelets_cluster_labels, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE of Wavelets with Clustering')
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        message = f'Index: {sel.target.index}'
        if additional_info:
            info = additional_info[sel.target.index]
            for key, value in info.items():
                message += f'\n{key}: {value}'
        sel.annotation.set_text(message)

    plt.show()