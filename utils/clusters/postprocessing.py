from inspect import currentframe
from itertools import combinations
import random
from typing import Callable

import numpy as np
from clusim.clustering import Clustering

from utils import generate_inputs


def get_sorted_clusters(clusters: list, metric: Callable[[list], tuple]) -> list:
    """
    Sort a list of clusters based on some metric
    :param clusters: The clusters
    :param metric: The metric for each cluster
    :return: The sorted list of clusters
    """
    # Compute the metric for the clusters
    sorting_list = [metric(cluster) for cluster in clusters]
    # Sort the clusters based on the metric
    zipped = list(zip(clusters, sorting_list))
    sorted_zipped = sorted(zipped, key=lambda entry: entry[1])
    return [cluster for cluster, _ in sorted_zipped]


def get_non_unique_membership_list(clusters) -> list:
    """
    Get the membership list for a cluster configuration containing non-unique clusters.
    If the same elements appear in multiple clusters, the first occurrence is returned
    :param clusters: The cluster configuration
    :return: The membership list for the cluster configuration
    """
    return [
        list(cluster_ids)[0]
        for element_id, cluster_ids
        in sorted(Clustering().from_cluster_list(clusters).to_elm2clu_dict().items())
    ]


def get_common_clusters(cluster_configurations_list: np.ndarray, mask: np.array = None) -> np.ndarray:
    """
    Find the most common sub-clusters among a list of cluster configurations
    :param cluster_configurations_list: The list of cluster configurations
    :param mask: The mask to filter the elements in the clusters
    :return: The sub-clusters and their counts, sorted by decreasing count and decreasing size
    """
    # Filter based on the mask
    if mask is not None:
        mask_idxs = np.argwhere(mask).flatten()
        masked_cluster_configurations = [
            [
                [
                    element for element in cluster if element in mask_idxs
                ]
                for cluster in cluster_configuration
            ]
            for cluster_configuration in cluster_configurations_list
        ]
    else:
        masked_cluster_configurations = cluster_configurations_list
    # Flatten the list of clusters, remove singletons
    masked_cluster_configurations = [
        cluster for cluster_configuration in masked_cluster_configurations for cluster in cluster_configuration
        if len(cluster) > 1
    ]
    # Find all the combinations of clusters
    combined = list(combinations(masked_cluster_configurations, 2))
    # Find all the possible intersections between the clusters
    intersections = [set(lhs).intersection(set(rhs)) for lhs, rhs in combined]
    # Remove the intersections of one element
    intersections = [intersection for intersection in intersections if len(intersection) > 1]
    # Count the occurrences of the intersections
    intersections_counts = list(zip(*np.unique(intersections, return_counts=True)))
    # Remove the intersections occurring only once
    intersections_counts = [
        (list(intersection), int(count))
        for intersection, count in intersections_counts
        if count > 1
    ]
    # Sort the intersections by decreasing count and decreasing size
    intersections_counts = sorted(intersections_counts, key=lambda entry: (-entry[1], -len(entry[0])))

    return np.array(intersections_counts)


def get_misclassified_items(cluster: np.ndarray):
    """
    Filter a cluster for the misclassified entries of the expected label
    :param cluster: The cluster to filter
    :return: The filtered cluster
    """
    # Get the indexes for the misclassified elements of the label
    mask_label = np.array(generate_inputs.test_labels == generate_inputs.EXPECTED_LABEL)
    mask_miss = np.array(generate_inputs.test_labels != generate_inputs.predictions)
    mask_idxs = np.argwhere(mask_miss[mask_label]).flatten()
    intersection = np.intersect1d(cluster, mask_idxs)
    return intersection


def select_data_by_cluster(df, train_data, train_labels, k):
    selected_indices = set()
    selected_data = []
    selected_lables = []
    
    for column in df[['clusters']]:
        columnSeriesObj = df[column]
        # there is only one row
        clusters = columnSeriesObj.values[0]

    print("Clusters:")
    print(len(clusters))
    backup_array = clusters
    while len(selected_indices) < k:
        for cluster in backup_array:
            # randomly select an element from each cluster
            if len(cluster) > 0:
                cluster = list(cluster)
                selected_index = random.choice(cluster)
                selected_indices.add(selected_index)
                cluster.remove(selected_index)
            if len(selected_indices) >= k:
                break

    remaining_data = []
    remaining_labels = []
    remaining_indices = []

    for i in range(0, len(train_data)):
        if i in selected_indices:
            selected_data.append(train_data[i])
            selected_lables.append(train_labels[i])
        else:
            remaining_data.append(train_data[i])
            remaining_labels.append(train_labels[i])
            remaining_indices.append(i)

    return np.array(selected_data), np.array(selected_lables), remaining_data, remaining_labels, remaining_indices


def select_data_by_cluster_both(HM_df, FM_df, train_data, train_labels, k):
    selected_indices = set()
    selected_data = []
    selected_lables = []

    for column in FM_df[['clusters']]:
            columnSeriesObj = FM_df[column]
            # there is only one row
            clusters_FM = columnSeriesObj.values[0]

    for column in HM_df[['clusters']]:
            columnSeriesObj = FM_df[column]
            # there is only one row
            clusters_HM = columnSeriesObj.values[0]


    while len(selected_indices) < k:
        not_added = True
        while not_added:
            for cluster in clusters_FM:
                cluster = list(cluster)
                if len(cluster) > 0:
                    selected_index = random.choice(cluster)
                    if selected_index not in selected_indices:
                        selected_indices.add(selected_index)
                        cluster.remove(selected_index)
                        not_added = False
                        break
                    else:
                        cluster.remove(selected_index)
            not_added = False
        
        not_added = True
        while not_added:
            for cluster in clusters_HM:
                cluster = list(cluster)
                if len(cluster) > 0:
                    selected_index = random.choice(cluster)
                    if selected_index not in selected_indices:
                        selected_indices.add(selected_index)
                        cluster.remove(selected_index)
                        not_added = False
                        break
                    else:
                        cluster.remove(selected_index)
            not_added = False
        
        not_added = True
        if len(selected_indices) >= k:
            break

    remaining_data = []
    remaining_labels = []
    remaining_indices = []

    for i in range(0, len(train_data)):
        if i in selected_indices:
            selected_data.append(train_data[i])
            selected_lables.append(train_labels[i])
        else:
            remaining_data.append(train_data[i])
            remaining_labels.append(train_labels[i])
            remaining_indices.append(i)

    return np.array(selected_data), np.array(selected_lables), remaining_data, remaining_labels, remaining_indices    