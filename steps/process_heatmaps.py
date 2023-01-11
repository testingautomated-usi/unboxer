import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import os.path
import warnings
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf

from config.config_data import classifier
from config.config_dirs import BEST_CONFIGURATIONS, HEATMAPS_DATA_RAW, \
    HEATMAPS_DATA
from config.config_heatmaps import APPROACH, CLUSTERING_TECHNIQUE, EXPLAINERS, DIMENSIONALITY_REDUCTION_TECHNIQUES, ITERATIONS
from utils.clusters.Approach import OriginalMode, GlobalLatentMode, LocalLatentMode
from utils.clusters.compare import compare_approaches
from utils.dataframes.sample import sample_highest_score
from IPython.display import display

APPROACHES = [OriginalMode, LocalLatentMode, GlobalLatentMode]


def get_perplexity(app):
    try:
        return app.get_dimensionality_reduction_techniques()[-1].get_params()['perplexity']
    except IndexError:
        return np.nan

def get_n_components(app):
    try:
        return app.get_dimensionality_reduction_techniques()[-1].get_params()['n_components']
    except IndexError:
        return np.nan

def main():
    # Ignore warnings from tensorflow
    warnings.filterwarnings('ignore')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Collect the approaches to use
    print('Collecting the approaches ...')
    # Select the approach from the configurations
    approach = APPROACHES[APPROACH]
    # Select the dimensionality reduction techniques based on the approach
    dimensionality_reduction_techniques = [[]] if approach is OriginalMode else DIMENSIONALITY_REDUCTION_TECHNIQUES
    # Collect the approaches
    approaches = [
        approach(
            explainer=explainer(classifier),
            dimensionality_reduction_techniques=dimensionality_reduction_technique
        )
        for explainer, dimensionality_reduction_technique
        in product(EXPLAINERS, dimensionality_reduction_techniques)
    ]
    # The processing mode is not original and there is a log -> read the configurations from the log
    # else:
    #     # Read the data about the best configurations
    #     best_configurations = pd.read_pickle(BEST_CONFIGURATIONS)
    #     # Find the best settings for each approach
    #     approaches = []
    #     for explainer in EXPLAINERS:
    #         try:
    #             # Find the best configuration to use
    #             best_config = best_configurations[
    #                  (best_configurations['approach'] == explainer.__qualname__)
    #                  & (best_configurations['clustering_mode'] == approach.__qualname__)
    #                  & (best_configurations[
    #                         'clustering_technique'] == CLUSTERING_TECHNIQUE.__qualname__)
    #                  ].iloc[0]
    #             approaches.append(
    #                 approach(
    #                     explainer=explainer(global_values.classifier),
    #                     dimensionality_reduction_techniques=best_config['dimensionality_reduction_techniques']
    #                 )
    #             )
    #         except IndexError:
    #             # Collect the approaches
    #             for approach in [
    #                 approach(
    #                     explainer=explainer(global_values.classifier),
    #                     dimensionality_reduction_techniques=dimensionality_reduction_technique
    #                 )
    #                 for explainer, dimensionality_reduction_technique
    #                 in product([explainer], dimensionality_reduction_techniques)
    #             ]:
    #                 approaches.append(approach)

    # Collect the data for the approaches
    df_raw = compare_approaches(
        approaches=approaches,
        iterations=ITERATIONS,
        get_info=lambda app: f"perplexity: {get_perplexity(app)}, dimensions: {get_n_components(app)}" if get_perplexity(app) != np.nan else "Original Mode"
    )
    # Export the raw data
    df_raw.to_pickle(HEATMAPS_DATA_RAW)

    df_sampled = df_raw

    if approach is not OriginalMode:
        # Find the best configuration for each explainer
        df_sampled = sample_highest_score(df_raw)
        best_configurations_tuples = list(
            df_sampled[['approach', 'dimensionality_reduction_techniques']].itertuples(index=False, name=None)
        )
        # Filter the dataset for the entries corresponding to the best configuration
        df_sampled = df_raw[df_raw[
            ['approach', 'clustering_mode', 'clustering_technique','dimensionality_reduction_techniques']].apply(tuple, axis=1).isin(best_configurations_tuples)
        ]
        # Remove the rows where the score is none -> black hole clusters
        df_sampled = df_sampled.dropna(subset=['score'])
            
        # Export the best configurations
        best_configs_df = df_sampled[
            ['approach', 'dimensionality_reduction_techniques']
        ].groupby('approach').first().reset_index(drop=False)
        best_configs_df.to_pickle(BEST_CONFIGURATIONS)

    # Export the sampled data
    df_sampled.to_pickle(HEATMAPS_DATA)
    return df_sampled


if __name__ == '__main__':
    main()
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    data = pd.read_pickle(HEATMAPS_DATA_RAW)
    display(data[['approach','dimensionality_reduction_techniques', 'score']])

