import math

import numpy as np
import seaborn as sns
from clusim.clustering import Clustering
from matplotlib import pyplot as plt

from config.config_outputs import MAX_LABELS, MAX_SAMPLES


def visualize_clusters_projections(
        projections: np.ndarray,
        cluster_membership: np.ndarray,
        aspect_ratio=4 / 3,
        palette='husl'
):
    """
    Visualize the projections for the clusters as a points
    :param projections: The 2D projections for the images
    :param cluster_membership: The cluster membership list for the projections
    :param aspect_ratio: The aspect ratio of the image
    :param palette: The color palette for the data points
    :return: The image and the axis
    """
    # mask_label = np.array(global_values.test_labels == global_values.EXPECTED_LABEL)
    # mask_miss = np.array(global_values.test_labels != global_values.predictions)
    # Create the figure
    fig = plt.figure(figsize=(10, 10 * aspect_ratio ** -1))
    # Plot the data
    ax = sns.scatterplot(
        x=[projection[0] for projection in projections],
        y=[projection[1] for projection in projections],
        hue=cluster_membership,
        # style=['misclassified' if is_masked else 'correct' for is_masked in mask_miss[mask_label]],
        palette=sns.color_palette(palette, n_colors=len(np.unique(cluster_membership)))
    )
    # Style
    sns.despine(left=True, bottom=True)
    ax.tick_params(left=False, bottom=False)
    ax.set(xticklabels=[], yticklabels=[])
    ax.legend(ncol=5)
    return fig, ax


def visualize_clusters_images(
        cluster_membership: np.ndarray,
        images: np.ndarray,
        predictions: np.ndarray,
        overlay: np.ndarray = None
):
    """
    Visualize a sample of clusters and images for each cluster as a grid
    :param cluster_membership: The cluster membership list
    :param images: The images for the entries in the cluster
    :param predictions: The predictions for the entries in the cluster
    :param overlay: The overlays for the entries in the clusters
    :return: The figure and the axis for the image
    """
    # Sample the clusters labels
    clusters_labels = np.unique(cluster_membership)
    clusters_membership_sample = cluster_membership
    images_sample = images
    predictions_sample = predictions
    overlay_sample = overlay if overlay is not None else np.empty_like(images)
    if clusters_labels.shape[0] > MAX_LABELS:
        # Sample the labels and get the corresponding mask
        sample_labels = np.random.choice(clusters_labels, MAX_LABELS, replace=False)
        labels_sample_mask = np.isin(cluster_membership, sample_labels)
        # Sample the clusters, images, predictions, and overlay based on the label
        clusters_membership_sample = cluster_membership[labels_sample_mask]
        images_sample = images[labels_sample_mask]
        predictions_sample = predictions[labels_sample_mask]
        overlay_sample = overlay[labels_sample_mask]

    # If the sample is empty (no misclassified image in the cluster) -> return empty image
    if len(clusters_membership_sample) == 0:
        fig, ax = plt.subplots(1, 1)
        ax.text(.5, .5, 'EMPTY SAMPLE', horizontalalignment='center', verticalalignment='center')
        return fig, ax
    # Find the labels in the sample and the number of entries for each label
    sample_labels, sample_labels_entries_count = np.unique(clusters_membership_sample, return_counts=True)
    # Add one row to prevent errors and remove it at the end
    n_rows = sample_labels.shape[0] + 1
    # The number of columns is the maximum number of entries for the labels + 1 for the label name
    n_cols = 1 + min(MAX_SAMPLES, np.amax(sample_labels_entries_count))

    # Create the image
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    # Keep track of the current image for the legend
    last_overlay = None
    # Plot the sample of images for each label
    for row, label in enumerate(sample_labels):
        # Show the labels name in the fist column
        ax[row][0].text(.5, .5, label, horizontalalignment='center', verticalalignment='center')
        # Filter the images, predictions, and overlay for the selected cluster label
        cluster_label_mask = clusters_membership_sample == label
        images_sample_label = images_sample[cluster_label_mask]
        predictions_sample_label = predictions_sample[cluster_label_mask]
        overlay_sample_label = overlay_sample[cluster_label_mask]
        # The number of entries is greater than the threshold -> sample
        images_sample_label_sample = images_sample_label
        predictions_sample_label_sample = predictions_sample_label
        overlay_sample_label_sample = overlay_sample_label
        if images_sample_label.shape[0] > MAX_SAMPLES:
            # Get the indexes for the random sample
            entries_sample_idxs = np.random.choice(images_sample_label.shape[0], MAX_SAMPLES, replace=False)
            # Filter the images, predictions, and overlay for the sample indexes
            images_sample_label_sample = images_sample_label[entries_sample_idxs]
            predictions_sample_label_sample = predictions_sample_label[entries_sample_idxs]
            overlay_sample_label_sample = overlay_sample_label[entries_sample_idxs]

        # Show the selected data
        zipped = zip(images_sample_label_sample, predictions_sample_label_sample, overlay_sample_label_sample)
        for col, elements in enumerate(zipped):
            image, prediction, overlay_image = elements
            # Show the image
            ax[row][col + 1].imshow(np.ma.masked_equal(image, 0).filled(np.nan), cmap='gray_r')
            # Show the overlay
            if overlay is not None:
                last_overlay = ax[row][col + 1].imshow(
                    np.ma.masked_equal(overlay_image, 0).filled(np.nan),
                    cmap='Reds',
                    alpha=.7
                )
            # Set the title
            ax[row][col + 1].set_title(f'Prediction: {int(prediction)}')

    # Remove ticks and labels
    try:
        ax_list = ax.flatten()
    except AttributeError:
        ax_list = [ax]
    for axx in ax_list:
        axx.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    # Add the legend if overlays where provided
    if last_overlay is not None:
        fig.subplots_adjust(right=0.89)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(last_overlay, cax=cbar_ax)
        cbar.set_ticks([])
    # Remove the last row
    [fig.delaxes(axx) for axx in ax[-1]]

    return fig, ax