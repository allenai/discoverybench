import click
import json
import os, fnmatch
import os.path
import csv
import sys

from new_eval import run_eval_gold_vs_gen_NL_hypo_workflow


def validate_query(ctx, param, query: str):
    if query == "":
        raise click.BadParameter("Query cannot be empty")
    return query


def validate_option(ctx, param, value):
    if value == "":
        raise click.BadParameter(f"Value {param.name} cannot be empty")
    return value


# @click.command()
# @click.option('--gold_hypo', type=str, callback=validate_option,  help='Gold Hypothesis', required=True)
# @click.option('--gold_workflow', type=str, help='Gold Workflow')
# @click.option('--pred_hypo', type=str, callback=validate_option, help='Predicted Hypothesis', required=True)
# @click.option('--pred_workflow', type=str, help='Predicted Workflow')
# @click.option('--metadata_path', help='Metadata file path', required=True)
# @click.option('--metadata_type', type=click.Choice(['real', 'synth']), help='Metadata type', required=True)
# @click.option('--eval_output_path', default="eval_output.json", help='Evaluation output path')
# @click.argument('query', type=str, callback=validate_query)
def evaluation(
    gold_hypo: str,
    pred_hypo: str,
    metadata_path: str,
    metadata_type: str,
    eval_output_path: str,
    query_id: str,
    gold_workflow="",
    pred_workflow="",
):
    query_rec = None
    try:
        with open(metadata_path, 'r') as f:
            data_metadata = json.load(f)
            if metadata_type == "synth":
                
                for rec in data_metadata['queries']:
                    print(f"rec: {rec}")
                    if int(rec["qid"]) == int(query_id):
                        query_rec = rec
            else:
                query_rec = data_metadata['queries'][0][query_id]
    except:
        data_metadata = None
    print(f"data_metadata: {data_metadata}")

    if query_rec == None:
        print(f"Query record not found: metadata_path={metadata_path}, qid:{qid}")
        return

    eval_result = run_eval_gold_vs_gen_NL_hypo_workflow(
        query=query_rec['question'],
        gold_hypo=gold_hypo,
        gold_workflow=gold_workflow,
        gen_hypo=pred_hypo,
        gen_workflow=pred_workflow,
        dataset_meta=data_metadata,
        llm_used='gpt-4-1106-preview',
        dataset_type=metadata_type,
        use_column_metadata=True,
    )

    print(json.dumps(eval_result, indent=4))
    f_readable = open(eval_output_path.replace("eval.", "eval_readable."), 'w')

    try:
        with open(eval_output_path, 'w') as f:
            f.write(json.dumps(eval_result))
            f_readable.write(json.dumps(eval_result, indent=4))
            # f.write(json.dumps(eval_result, indent=4))
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    #evaluation()
    args = sys.argv[1:]

    print(f"args: {args}")
    offset = 0
    if args[0] == 'python':
        offset = 2
    folder_name = args[offset+0]
    dataset_type = args[offset+1]

    # parsed_folder = "outputs/my_logs/react_agent-llama-3/eval/"
    # output_folder = "outputs/my_logs/react_agent-llama-3/eval_new/"
    
    parsed_folder = f"outputs/my_logs/{folder_name}/eval/"
    output_folder = f"outputs/my_logs/{folder_name}/eval_new/"

    # parsed_folder = "outputs/my_logs/May27_DV_gpt-4-0125-preview/eval/"
    # output_folder = "outputs/my_logs/May27_DV_gpt-4-0125-preview/eval_new/"

    if dataset_type == "real":
        answer_key_path = "eval/answer_key_real.csv"
    else:
        answer_key_path = "eval/answer_key_synth.csv"
    answer_key_file = csv.reader(open(answer_key_path,"r"))
    dataset_metaid_qid_to_gold_hypo = dict()
    for row in answer_key_file:
        dataset = row[0]
        meta_id = row[1]
        qid = row[2]
        gold_hypo = row[3]
        uniq_id = f"{dataset}||{meta_id}||{qid}"
        dataset_metaid_qid_to_gold_hypo[uniq_id] = gold_hypo

    for file_name in fnmatch.filter(os.listdir(parsed_folder), 'eval.*.json'):
        if "(1)" in file_name:
            continue
        
        output_path = output_folder + "/" +file_name.replace("log", "eval")
    
        parts = file_name.replace("log.", "").replace("eval.", "").replace(".json", "").split(".")
        if len(parts) < 3 or (parts[2] == "fixed"):
            continue

        dataset = parts[0]
        if dataset == "meta_regression_processed":
            dataset = "meta_regression"
        elif dataset == "nls_incarceration_processed":
            dataset = "nls_incarceration"
        elif dataset == "nls_ses_processed":
            dataset = "nls_ses"
        elif dataset == "nls_bmi_processed":
            dataset = "nls_bmi"

        metadata = parts[1]
        query_id = parts[2]

        if dataset_type == "real":
            dataset_base_path = "discoverybench/real/train_and_test"
        else:
            dataset_base_path = "discoverybench/synth/test"
        metadata_path = f"{dataset_base_path}/{dataset}/metadata_{metadata}.json"

        key = file_name.replace("log.", "").replace(".json", "")\
            .replace(".", "||").replace("meta_regression_processed", "meta_regression")\
            .replace("nls_incarceration_processed", "nls_incarceration")\
            .replace("nls_ses_processed", "nls_ses")\
            .replace("nls_bmi_processed", "nls_bmi")
        print(f"key: {key}")

        if os.path.exists(output_path):
            if os.path.getsize(output_path) > 0: #skip the queries that are already evaluated
                print(f"skipping query :{key} ")
                continue

        print(f"metadata_path: {metadata_path}")

        if key in dataset_metaid_qid_to_gold_hypo:
            gold_hypo = dataset_metaid_qid_to_gold_hypo[key]
        else:
            if not os.path.exists(metadata_path):
                print(f"Metadata not found")
                continue

            # try:
            #     with open(metadata_path, 'r') as f:
            #         data_metadata = json.load(f)
            # except:
            #     data_metadata = None
            # # gold_hypo = data_metadata['queries'][0][int(query_id)]['true_hypothesis']
            
        prediction_str = open(parsed_folder + "/" + file_name, "r").readline()
        # print(f"file_name: {parsed_folder}/{file_name}\n prediction_str: {prediction_str}")
        prediction_json = json.loads(prediction_str)
        # print(f"\nprediction_json: {prediction_json}")
        gold_hypo = prediction_json['HypoA']
        gold_workflow = prediction_json['WorkflowA']

        # {"final_answer": " `Given the observed error, ...."
        # "workflow_summary": .....}
        result = evaluation(gold_hypo=gold_hypo,
                            pred_hypo=prediction_json['HypoB'],
                            metadata_path=metadata_path,
                            metadata_type="synth",
                            eval_output_path=output_path,
                            query_id=query_id,
                            # gold_workflow=gold_workflow,
                            # pred_workflow=prediction_json['WorkflowB']
                            )
        
        
        



