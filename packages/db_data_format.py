from typing import Iterable
import polars as pl
import json
import pandas as pd
import os

from .agent_dataclasses import DBDataset

def load_db_dataframe(project: str, db_path: str = "discoverybench/real/train") -> pl.DataFrame:
    rows = []
    complete_path = f"{db_path}/{project}"
    for fname in filter(lambda x: x.endswith('json'), os.listdir(complete_path)):
        with open(f"{complete_path}/{fname}", 'r') as file:
            try:
                metadata = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error occurred: {fname}.")
                raise e
            assert len(metadata['queries']) == 1
            for query_dict in metadata['queries'][0]:
                qid = query_dict['qid']
                query = query_dict['question']
                hypothesis = query_dict['true_hypothesis']
                workflow = metadata['workflow']
                metadata_id = int(fname.split(".")[0].split("_")[1])
                metadata_path = f"{complete_path}/{fname}"
                rows.append([project, 
                             metadata_id, 
                             qid, 
                             hypothesis])
                # TODO: randomly assign the `is_test` column.
    return pl.DataFrame({
        'dataset': [x[0] for x in rows],
        'metadataid': [x[1] for x in rows],
        'query_id': [x[2] for x in rows],
        'gold_hypo': [x[3] for x in rows]
    })

def construct_db_dataset_objs(metadata_path) -> Iterable[DBDataset]:
    db_datasets = []
    with open(metadata_path, 'r') as metadata_file: 
        meta_dataset = json.load(metadata_file)
        for dataset in meta_dataset['datasets']:
            column_names = []
            column_descriptions = []
            spreadsheet_name = dataset['name']
            spreadsheet_path = os.path.join(os.path.dirname(metadata_path), spreadsheet_name)
            if spreadsheet_name.endswith('.csv'):
                num_datapoints = len(pd.read_csv(spreadsheet_path)) 
                dtypes = pd.read_csv(spreadsheet_path).dtypes.to_dict()
                dtypes = [dtypes[column] for column in dtypes]
            elif spreadsheet_name.endswith('.tsv'):
                num_datapoints = len(pd.read_csv(spreadsheet_path, sep='\t'))
                dtypes = pd.read_csv(spreadsheet_path, sep='\t').dtypes.to_dict()
                dtypes = [dtypes[column] for column in dtypes]
            else:
                raise ValueError(f"Unrecognized file extension for {spreadsheet_name}")
            for column in dataset['columns']['raw']:
                column_names.append(column['name'])
                column_descriptions.append(column['description'])
            db_datasets.append(DBDataset(num_datapoints, column_names, column_descriptions, dtypes, spreadsheet_path))
    return db_datasets

def describe_db_dataset(db_dataset):
    column_explanations = "\n".join([
        f"{i}. '{column}': {db_dataset.column_explanations[i]} (dtype: {db_dataset.column_datatypes[i]})" for i, column in enumerate(db_dataset.columns)
    ])
    return f"Dataset location: {db_dataset.spreadsheet_path}\n" +\
        f"Number of datapoints: {db_dataset.num_datapoints}\n" +\
        f"Columns descriptions:\n{column_explanations}"

def generate_variable_exploration_prompt(metadata_path, query):
    db_datasets = construct_db_dataset_objs(metadata_path)
    dataset_description_str = "\n-------------------\n".join([describe_db_dataset(db_dataset) for db_dataset in db_datasets])

    return f"Consider the following query: {query}" +\
        f" It can be answered by analyzing the following datasets:\n-------------------\n" +\
        f"{dataset_description_str}\n-------------------\n" +\
        " What do you think are the relevant columns from the dataset(s) for answering the query?" +\
        f" Write some code to explore the distributions of only those columns." +\
        f" For continuous variables, consider the 5-number summary (median, 1st quartile, 3rd quartile, min, max)." +\
        f" For categorical variables, consider the frequency of each category (both absolute and relative)." +\
        f" Do not try to answer the question yet. Don't draw any plots and don't use df.info."