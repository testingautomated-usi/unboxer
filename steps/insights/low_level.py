from config.config_data import EXPECTED_LABEL
import numpy as np
import pandas as pd
import seaborn as sns
from clusim.clustering import Clustering
from matplotlib import pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from config.config_dirs import HEATMAPS_DATA, HEATMAPS_DATA_RAW
from config.config_general import CLUSTERS_SORT_METRIC, CLUSTERS_SIMILARITY_METRIC
from utils import generate_inputs
from utils.clusters.postprocessing import get_sorted_clusters
from utils.dataframes.sample import sample_most_popular
from utils.general import save_figure
from utils.plotter.distance_matrix import show_comparison_matrix
from utils.plotter.visualize import visualize_clusters_projections, visualize_clusters_images


def heatmaps_distance_matrix():
    # Read the data
    df = pd.read_pickle(HEATMAPS_DATA)
    # Remove the configurations with only one clusters
    df['num_clusters'] = df['clusters'].apply(len)
    plot_data = df[df['num_clusters'] > 1]
    distances_df, fig, ax = show_comparison_matrix(
        values=[Clustering().from_cluster_list(clusters) for clusters in plot_data['clusters']],
        metric=lambda lhs, rhs: 1 - CLUSTERS_SIMILARITY_METRIC(lhs, rhs),
        index=plot_data['approach'],
        show_progress_bar=True,
        remove_diagonal=False
    )
    fig.suptitle('Distance matrix for the low-level approaches')
    save_figure(fig, f'out/low_level/distance_matrix')


def heatmaps_clusters_projections():
    # Read the data
    df = pd.read_pickle(HEATMAPS_DATA)
    # Get the most popular configurations
    df = sample_most_popular(df)
    # Iterate through the explainers
    approaches = df.index.unique()
    for approach in tqdm(approaches, desc='Exporting the clusters projections'):
        # Get the best configuration for the explainer
        pick_config = df.loc[approach]
        clusters, projections, contributions = pick_config[[
            'clusters',
            'projections',
            'contributions'
        ]]
        # Convert the clusters to membership list
        clusters_membership = np.array(Clustering().from_cluster_list(clusters).to_membership_list())
        # Visualize the projections of the contributions clusters
        fig, ax = visualize_clusters_projections(projections=projections, cluster_membership=clusters_membership)
        ax.set_title(f'{approach} clusters projections')
        save_figure(fig, f'out/low_level/{approach}/clusters_projections')


def heatmaps_clusters_images():
    # Read the data
    df = pd.read_pickle(HEATMAPS_DATA)
    # Get the most popular configurations
    df = sample_most_popular(df)

    _, _, test_data, test_labels, predictions = generate_inputs.load_inputs()
    test_data_gs = tf.image.rgb_to_grayscale(test_data)

    mask_label = np.array(test_labels == EXPECTED_LABEL)
    mask_miss = np.array(test_labels != predictions)
    approaches = df.index.unique()
    for approach in tqdm(approaches, desc='Exporting the clusters sample images'):
        # Get the best configuration for the explainer
        pick_config = df.loc[approach]
        clusters, projections, contributions = pick_config[[
            'clusters',
            'projections',
            'contributions'
        ]]
        # Convert the clusters to membership list
        clusters_membership = np.array(Clustering().from_cluster_list(clusters).to_membership_list())
        # Get the mask for the clusters containing misclassified elements of the selected label
        mask_contains_miss_label = np.isin(
            clusters_membership,
            np.unique(clusters_membership[mask_miss[mask_label]])
        )

        # Sample some clusters labels containing misclassified items
        clusters_labels = np.unique(clusters_membership[mask_contains_miss_label])
        sample_labels = np.random.choice(clusters_labels, min(4, len(clusters_labels)), replace=False)
        sample_mask = np.isin(clusters_membership, sample_labels)
        # Sort the clusters if a sorting parameter is provided
        if CLUSTERS_SORT_METRIC is not None:
            clusters = get_sorted_clusters(clusters, metric=CLUSTERS_SORT_METRIC)
            clusters_membership = np.array(Clustering().from_cluster_list(clusters).to_membership_list())
        # Show some correctly classified images for clusters containing also misclassified images
        correct_sample_mask = mask_contains_miss_label & ~mask_miss[mask_label] & sample_mask
        fig, _ = visualize_clusters_images(
            cluster_membership=clusters_membership[correct_sample_mask],
            images=test_data_gs[mask_label][correct_sample_mask],
            predictions=predictions[mask_label][correct_sample_mask],
            overlay=contributions[correct_sample_mask]
        )
        save_figure(fig, f'out/low_level/{approach}/clusters_correct_images')
        # Show some incorrectly classified images for clusters containing also misclassified images
        misses_sample_mask = mask_contains_miss_label & mask_miss[mask_label] & sample_mask
        fig, _ = visualize_clusters_images(
            cluster_membership=clusters_membership[misses_sample_mask],
            images=test_data_gs[mask_label][misses_sample_mask],
            predictions=predictions[mask_label][misses_sample_mask],
            overlay=contributions[misses_sample_mask]
        )
        save_figure(fig, f'out/low_level/{approach}/clusters_misclassified_images')


def heatmaps_silhouette_by_perplexity():
    # Read the data
    df = pd.read_pickle(HEATMAPS_DATA_RAW)
    # Iterate through the explainers
    approaches = df['approach'].unique()
    for explainer in tqdm(approaches, desc='Showing the distribution of the silhouette score'):
        # Filter the dataframe for the explainer
        explainer_df = df[df['approach'] == explainer]
        # Show the distribution of the silhouette by perplexity
        fig = plt.figure(figsize=(16, 9))
        sns.boxplot(x='perplexity', y='silhouette', data=explainer_df, color='gray').set_title(
            f'{explainer} silhouette score by perplexity'
        )
        save_figure(fig, f'out/low_level/{explainer}/silhouette_by_perplexity')
