def get_dv_query_for_real(metadata, provide_domain_knowledge, provide_workflow_tags, nl_query):
    dataset_meta = ""
    for dataset_metadata in metadata['datasets']:
        dataset_meta += "Dataset name: " + dataset_metadata['name']
        dataset_meta += "Dataset description: " + dataset_metadata['description']
        dataset_meta += "\nBrief description of columns: "
        for col in dataset_metadata['columns']['raw']:
            dataset_meta += col['name'] + ": " + col['description'] + ", "

    query_to_dv = dataset_meta

    for int_hypo in metadata['hypotheses']['intermediate']:
        query_to_dv += int_hypo['text'] + ",\n "

    query_to_dv += f"\nQuery: {nl_query}"

    if provide_domain_knowledge:
        query_to_dv += "\nAdditionally, we provide some hints that might be useful to solve the task. Domain Knowledge: \n" + metadata['domain_knowledge']+".\n"

    if provide_workflow_tags:
        query_to_dv += "The meta tags are: " + metadata['workflow_tags'] + ".\n"

    query_to_dv += "In the final answer, please write down a scientific hypothesis in "\
        "natural language, derived from the provided dataset, clearly stating the "\
        "context of hypothesis (if any), variables chosen (if any) and "\
        "relationship between those variables (if any) including any statistical significance."\
        "Also generate a summary of the full workflow starting from data loading that led to the final answer as WORKFLOW SUMMARY:"

    # Run the NL query through datavoyager
    print(f"query_to_dv: {query_to_dv}")
    return query_to_dv, dataset_meta


def get_dv_query_for_synth(metadata, nl_query):
    dataset_meta = ""
    for dataset_metadata in metadata['datasets']:
        dataset_meta += "Dataset name: " + dataset_metadata['name']
        dataset_meta += "Dataset description: " + dataset_metadata['description']
        dataset_meta += "\nBrief description of columns: "
        for col in dataset_metadata['columns']:
            dataset_meta += col['name'] + ": " + col['description'] + ", "

    query_to_dv = dataset_meta

    query_to_dv += f"\nQuery: {nl_query}"
    query_to_dv += "In the final answer, please write down a scientific hypothesis in "\
        "natural language, derived from the provided dataset, clearly stating the "\
        "context of hypothesis (if any), variables chosen (if any) and "\
        "relationship between those variables (if any) including any statistical significance."

    # Run the NL query through datavoyager 
    print(f"query_to_dv: {query_to_dv}")
    return query_to_dv, dataset_meta


def run_autonomous_single_agent_discoverybench(
    agent,
    datasets,
    metadata,
    nl_query,
    provide_domain_knowledge,
    provide_workflow_tags,
    dataset_type
):
    if dataset_type == "real":
        query_to_dv, dataset_meta = get_dv_query_for_real(
            metadata,
            provide_domain_knowledge,
            provide_workflow_tags,
            nl_query
        )
    else:
        query_to_dv, dataset_meta = get_dv_query_for_synth(
            metadata,
            nl_query
        )

    agent.generate(
        dataset_paths=datasets,
        query=query_to_dv,
    )
