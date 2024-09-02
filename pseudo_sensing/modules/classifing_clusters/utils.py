import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def save_cluster_class(cluster_labels, cluster_class, output_dir, rows, cols, save_fig=False):
    """
    Save the cluster class as a json file.
    
    Args:
    cluster_labels: ndarray, array containing the cluster labels
    cluster_class: dict, cluster class
    output_dir: str, path to save the json file
    rows: int, number of rows in the grid
    cols: int, number of columns in the grid
    save_fig: bool, save the figure
    """
    
    labels = []
    for label in cluster_labels:
        labels.append(cluster_class[label])
    
    labels_grid = {}
    for i in range(rows):
        for j in range(cols):
            labels_grid[f'{i}_{j}'] = labels[i * cols + j]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'cluster_class.json'), 'w') as f:
        json.dump(labels_grid, f, indent=4)
    
    if save_fig:
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(len(unique_labels) + 1) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        plt.figure(figsize=(8, 8))
        cluster_labels_reshaped = cluster_labels.reshape(rows, cols)
        plt.imshow(cluster_labels_reshaped, cmap=cmap, norm=norm)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(value)), markersize=10, label=f'{cluster_class[value]}')
            for value in unique_labels
        ]
        plt.legend(handles=handles, loc='upper right')
        plt.title('Cluster Class')
        plt.savefig(os.path.join(output_dir, 'cluster_class.png'))
        plt.close()

def copy_wavelet_to_each_cluster_dir(output_dir, wavelets_cluster_labels, cluster_class, rows, cols):
    """
    Copy the wavelet images to each cluster directory.
    
    Args:
    output_dir: str, path to the output directory
    wavelets_cluster_labels: ndarray, array containing the cluster labels
    cluster_class: dict, cluster class
    rows: int, number of rows in the grid
    cols: int, number of columns in the grid
    """
    clusters_output_dir = os.path.join(output_dir, "clusters")
    for cluster_label in np.unique(wavelets_cluster_labels):
        cluster_dir = os.path.join(clusters_output_dir, cluster_class[cluster_label])
        os.makedirs(cluster_dir, exist_ok=True)
        for i in range(rows):
            for j in range(cols):
                if wavelets_cluster_labels[i * cols + j] == cluster_label:
                    wavelet_file = os.path.join(output_dir, "wavelets", f"wavelet_{i}_{j}.png")
                    os.system(f'cp "{wavelet_file}" "{cluster_dir}"')