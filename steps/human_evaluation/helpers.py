from config.config_data import EXPECTED_LABEL
import numpy as np
import pandas as pd

from config.config_dirs import FEATUREMAPS_DATA, HEATMAPS_DATA, HEATMAPS_DATA_RAW
from utils import generate_inputs
from utils.dataframes.extractor import get_average_popularity_score
from IPython.display import display

def preprocess_featuremaps_data():
    # Read the featuremaps data
    try:
        df = pd.read_pickle(FEATUREMAPS_DATA)
    except FileNotFoundError:
        return pd.DataFrame()
    # Compute the total time
    df['time'] = df['map_time'] + df['features_extraction']
    df = df.drop(columns=['map_time', 'features_extraction'])
    # Get the overall information about the approach
    df['approach'] = df.apply(
        lambda row: f'{row["approach"]}({row["map_size"]})_{row["mode"]}',
        axis=1
    )
    df = df.drop(columns=['map_size', 'mode'])

    return df


def preprocess_heatmaps_data():
    # Read the heatmaps data
    try:
        df = pd.read_pickle(HEATMAPS_DATA_RAW)
    except ValueError:
        return pd.DataFrame()
    # Make the naming consistent with the featuremaps
    df = df.rename(columns={'explainer': 'approach'})
    df['time'] = df['time_clustering'] + df['time_contributions']
    pd.set_option('display.max_colwidth', None)
    display(df)
    df['clustering_mode'] = df.apply(lambda row: f'{row["clustering_technique"]}({row["clustering_mode"]})', axis=1)

    # Keep the column of interest
    df = df[
        ['approach', 'clustering_mode', 'time', 'clusters', 'contributions']]
    df = df.rename(columns={'explainer': 'approach'})
    
    return df


def preprocess_data():
    # Read all the needed data
    featuremaps_data = preprocess_featuremaps_data()
    heatmaps_data = preprocess_heatmaps_data()

    _, _, generated_data, generated_labels, generated_predictions = generate_inputs.load_inputs()

    mask_label = np.array(generated_labels == EXPECTED_LABEL)
    mask_miss = np.array(generated_labels != generated_predictions)
    misclassified_idxs = np.argwhere(mask_miss[mask_label])

    # Merge the data for the featuremaps and the heatmaps
    merged = pd.concat([heatmaps_data, featuremaps_data]).reset_index(drop=True)
    # Compute additional information
    merged['num_clusters'] = merged['clusters'].apply(len)
    merged['clusters_sizes'] = merged['clusters'].apply(lambda clusters: [len(cluster) for cluster in clusters])
    merged['frac_misses'] = merged['clusters'].apply(
        lambda clusters: [
            len([entry for entry in cluster if entry in misclassified_idxs]) / len(cluster)
            for cluster in clusters
        ]
    )
    merged['frac_mixed'] = merged['frac_misses'].apply(
        lambda misses: len([entry for entry in misses if 0 < entry < 1]) / len(misses)
    )
    # pd.set_option('display.max_colwidth', None)
    return merged


def sample_clusters():
    df = preprocess_data()
    # Find the average popularity  score for the configurations
    df['popularity_score'] = get_average_popularity_score(df)
    # For each approach, choose the cluster configuration with the highest popularity score
    df['rank'] = df.groupby('approach')['popularity_score'].rank(method='dense', ascending=False).astype(int)
    sampled_clusters = df[df['rank'] == 1].drop(columns='rank')
    sampled_clusters = sampled_clusters.groupby('approach').first().reset_index()

    return sampled_clusters
