from packages.db_data_format import describe_db_dataset, construct_db_dataset_objs, generate_variable_exploration_prompt

def test_db_dataset_objs_construction():
    metadata_path = "db_unsplit/introduction_pathways_non-native_plants/metadata_0.json"
    objs = construct_db_dataset_objs(metadata_path)
    assert objs[0].spreadsheet_path == "db_unsplit/introduction_pathways_non-native_plants/temporal_trends_contingency_table.tsv"
    assert objs[0].columns[0] == "introduction.period"
    assert objs[0].column_explanations[0] == "This column represents different time periods, related to when non-native plant species were introduced into the region."

def test_describe_dataset():
    metadata_path = "db_unsplit/introduction_pathways_non-native_plants/metadata_0.json"
    objs = construct_db_dataset_objs(metadata_path)
    print()
    print(describe_db_dataset(objs[0]))

def test_exploration_prompt():
    prompt = generate_variable_exploration_prompt("db_unsplit/introduction_pathways_non-native_plants/metadata_0.json", "What is the datatype of the column 'introduction.period'?")
    print()
    print(prompt)

