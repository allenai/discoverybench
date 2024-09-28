import click
import json
from eval.new_eval import run_eval_gold_vs_gen_NL_hypo_workflow


def validate_query(ctx, param, query: str):
    if query == "":
        raise click.BadParameter("Query cannot be empty")
    return query


def validate_option(ctx, param, value):
    if value == "":
        raise click.BadParameter(f"Value {param.name} cannot be empty")
    return value


@click.command()
@click.option('--gold_hypo', type=str, callback=validate_option,  help='Gold Hypothesis', required=True)
@click.option('--gold_workflow', type=str, help='Gold Workflow')
@click.option('--pred_hypo', type=str, callback=validate_option, help='Predicted Hypothesis', required=True)
@click.option('--pred_workflow', type=str, help='Predicted Workflow')
@click.option('--metadata_path', help='Metadata file path', required=True)
@click.option('--metadata_type', type=click.Choice(['real', 'synth']), help='Metadata type', required=True)
@click.option('--eval_output_path', default="eval_output.json", help='Evaluation output path')
@click.argument('query', type=str, callback=validate_query)
def evaluation(
    gold_hypo: str,
    gold_workflow: str,
    pred_hypo: str,
    pred_workflow: str,
    metadata_path: str,
    metadata_type: str,
    eval_output_path: str,
    query: str
):
    with open(metadata_path, 'r') as f:
        data_metadata = json.load(f)

    eval_result = run_eval_gold_vs_gen_NL_hypo_workflow(
        query=query,
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

    try:
        with open(eval_output_path, 'a') as f:
            f.write(json.dumps(eval_result, indent=4))
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    evaluation()
