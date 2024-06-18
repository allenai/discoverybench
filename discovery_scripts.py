from flowmason import SingletonStep, conduct, load_artifact_with_step_name
import pandas as pd
from collections import OrderedDict

CACHE_DIR="scratch/flowmason_cache"
LOG_DIR="discoveryagent_logs"

def load_dataset(file_path: str, **kwargs):
    data = pd.read_csv(file_path)
    return data

def compute_correlations(data: pd.DataFrame, **kwargs):
    correlation_matrix = data.corr()
    return correlation_matrix

def find_strongest_correlation(correlation_matrix: pd.DataFrame, target_variable: str, **kwargs):
    correlations = correlation_matrix[target_variable].drop(target_variable)
    strongest_factor = correlations.abs().idxmax()
    strongest_correlation = correlations[strongest_factor]
    return strongest_factor, strongest_correlation

step_dict = OrderedDict()
step_dict['load_dataset'] = SingletonStep(load_dataset, {
    'version': "001",
    'file_path': 'discoverybench/real/train/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv'
})

step_dict['compute_correlations'] = SingletonStep(compute_correlations, {
    'version': "001",
    'data': 'load_dataset'
})

step_dict['find_strongest_correlation'] = SingletonStep(find_strongest_correlation, {
    'version': "001",
    'correlation_matrix': 'compute_correlations',
    'target_variable': 'BAMM_speciation'
})

run_metadata = conduct(CACHE_DIR, step_dict, LOG_DIR)
strongest_factor, strongest_correlation = load_artifact_with_step_name(run_metadata, 'find_strongest_correlation')
print(strongest_factor, strongest_correlation)