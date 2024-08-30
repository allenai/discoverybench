from flowmason import conduct, MapReduceStep, SingletonStep, load_artifact_with_step_name
from .constants import MY_CACHE_DIR, MY_LOG_DIR
import click
import ipdb

@click.command()
@click.argument('dataset_path')
@click.option('--structure-type',
              type=click.Choice(['loose', 'functions', 'dag'], case_sensitive=False))
@click.option('--add_exploration_step', is_flag=True, help='Have the agent first explore the univariate distributions of the dataset.')
# @click.option('--self_consistency', is_flag=True) # keep this off for now.
@click.option('--meta_reason_samples', is_flag=True) # keep this off for now.
def execute_library_learning(dataset_path: str, structure_type, add_exploration_step: bool, 
                            #  self_consistency: bool, 
                             meta_reason_samples: bool):
    # steps:
        # 1. load all the metadata files
        # 2. write a map step to process each metadata file.
        # 3. Try to get it correct, otherwise provide supervision by applying the workflow

        # then, write another step to collect all the analyses. Make sure to track whether we got it correct on the first try or had to provide supervision.
        # then, write another step to push forward the correct analyses 
        # then, prompt the model to make the programs more efficient (?) 
    frame = construct_db_dataset_frame(dataset_path, "db_unsplit/answer_key_real.csv")
    train_frame = frame.filter(pl.col('is_test')==False)
    test_frame = frame.filter(pl.col('is_test')==True)

    ipdb.set_trace()
    # dataset_queries = frame['query'].to_list()
    # dataset_paths = frame['metadata_path'].to_list()
    # analysis_ids = frame['analysis_id'].to_list()
    # hypotheses = frame['gold_hypo'].to_list()
    # workflows = frame['workflow'].to_list()

    ll_steps_dict = OrderedDict()

    mapreduce_dict = OrderedDict()
    mapreduce_dict['step_complete_pass'] = SingletonStep(analyze_db_dataset, { # TODO: run the analysis with the agent options
        "version": "001"
    })
    mapreduce_dict['step_evaluate_analysis'] = SingletonStep(step_evaluate_claim, {
        "experiment_result": 'step_complete_pass',
        'version': '001'
    })
    mapreduce_dict['step_provide_supervision_on_incorrect'] = SingletonStep(step_provide_supervision_on_incorrect, { # TODO: have to implement this step
        'experiment_result': 'step_complete_pass',
        'evaluation': 'step_evaluate_analysis', 
        'self_consistency': True,
        'version': '003'
    })
    mapreduce_dict['step_rerun_evaluation_for_supervised_analysis'] = SingletonStep(step_evaluate_claim, { # must return a class with an experiment result, whether it's correct, and whether supervision was necessary
        'experiment_result': 'step_provide_supervision_on_incorrect',
        'version': '001'
    })
    mapreduce_dict['step_consolidate_results'] = SingletonStep(step_combine_analysis_with_evaluation, {
        'experiment_result': 'step_complete_pass',
        'claim_evaluation': 'step_evaluate_analysis',
        'supervised_result': 'step_provide_supervision_on_incorrect',
        'revised_evaluation': 'step_rerun_evaluation_for_supervised_analysis',
        'version': '001'
    })
    ll_steps_dict['map_reduce_train_program_generation'] = MapReduceStep(mapreduce_dict, 
        {
            "query": train_frame['query'].to_list(),
            'analysis_id': train_frame['analysis_id'].to_list(), 
            "dataset_path": train_frame['metadata_path'].to_list(),
            'gold_hypothesis': train_frame['gold_hypo'].to_list(),
            'workflow': train_frame['workflow'].to_list()
        },
        {
            'self_consistency': False, # NOTE: this is overriden for when we provide workflows. 
            'structure_type': structure_type,
            'add_exploration_step': add_exploration_step, 
            'generate_visual': False,
            'version': '001'
        },  list, 'analysis_id', []
    )
    ll_steps_dict['compute_metrics'] = SingletonStep(compute_metrics_db, {
        "experiment_result_claims": 'map_reduce_train_program_generation',
        "version": "001",
    })
    ll_steps_dict['compute_library_fns']= SingletonStep(create_ll_prompt, {
        'queries': tuple(train_frame['query'].to_list()),
        'experiment_result_claims': 'map_reduce_train_program_generation',
        'version': '006', 
        'meta_reason_samples': meta_reason_samples
    })
    evaluation_mapreduce_dict = OrderedDict()
    evaluation_mapreduce_dict['step_complete_pass_w_library'] = SingletonStep(analyze_db_dataset, { 
        "version": "002",
        'library_str': 'compute_library_fns'
    })
    evaluation_mapreduce_dict['step_evaluate_analysis_w_library'] = SingletonStep(step_evaluate_claim, {
        "experiment_result": 'step_complete_pass_w_library',
        'version': '001'
    })
    evaluation_mapreduce_dict['step_consolidate_results_w_library'] = SingletonStep(step_combine_analysis_with_evaluation_test, {
        'experiment_result': 'step_complete_pass_w_library',
        'claim_evaluation': 'step_evaluate_analysis_w_library',
        'version': '001'
    })
    ll_steps_dict['map_reduce_test_w_library'] = MapReduceStep(evaluation_mapreduce_dict, 
        {
            "query": test_frame['query'].to_list(),
            'analysis_id': test_frame['analysis_id'].to_list(), 
            "dataset_path": test_frame['metadata_path'].to_list(),
            'gold_hypothesis': test_frame['gold_hypo'].to_list()
        },
        {
            'self_consistency': False,
            'structure_type': structure_type,
            'add_exploration_step': add_exploration_step, 
            'generate_visual': False,
            'version': '001'
        },  list, 'analysis_id', []
    )
    ll_steps_dict['compute_metrics_on_library_analyses'] = SingletonStep(compute_metrics_db, { 
        "experiment_result_claims": 'map_reduce_test_w_library',
        'version': '001'
    })
    evaluation_no_library_dict = OrderedDict()
    evaluation_no_library_dict['step_complete_pass_no_library'] = SingletonStep(analyze_db_dataset, {
        'version': '002',
    })
    evaluation_no_library_dict['step_evaluate_analysis_no_library'] = SingletonStep(step_evaluate_claim, {
        "experiment_result": 'step_complete_pass_no_library',
        'version': '001'
    })
    evaluation_no_library_dict['step_consolidate_results_no_library'] = SingletonStep(step_combine_analysis_with_evaluation_test, {
        'experiment_result': 'step_complete_pass_no_library',
        'claim_evaluation': 'step_evaluate_analysis_no_library',
        'version': '001'
    })
    ll_steps_dict['map_reduce_test_no_library'] = MapReduceStep(evaluation_no_library_dict,
        {
            "query": test_frame['query'].to_list(),
            'analysis_id': test_frame['analysis_id'].to_list(), 
            "dataset_path": test_frame['metadata_path'].to_list(),
            'gold_hypothesis': test_frame['gold_hypo'].to_list()
        },
        {
            'self_consistency': False,
            'structure_type': structure_type,
            'add_exploration_step': add_exploration_step, 
            'generate_visual': False,
            'version': '002'
        },  list, 'analysis_id', []
    )
    ll_steps_dict['compute_metrics_on_no_library_analyses'] = SingletonStep(compute_metrics_db, { 
        "experiment_result_claims": 'map_reduce_test_no_library',
        'version': '001'
    })
    run_metadata = conduct(MY_CACHE_DIR, ll_steps_dict, MY_LOG_DIR)
    output_library_fn = load_artifact_with_step_name(run_metadata, 'compute_library_fns')
    print(output_library_fn)
    ipdb.set_trace()
