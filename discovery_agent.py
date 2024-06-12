import click
import os
import json
from agents.coder_agent import BaseAgent
from agents.react_agent import ReactAgent
from utils.autonomous_single_agent import run_autonomous_single_agent_discoverybench

# set the environment variables
os.environ['MODEL_CONFIG'] = 'config/model_config.json'
os.environ['API_CONFIG'] = 'config/api_config.json'


def validate_model_name(ctx, param, model_name: str):
    with open('config/model_config.json', 'r') as file:
        model_config = json.load(file)
    if model_name not in model_config['models']:
        raise click.BadParameter(f"Model {model_name} not found in model config")
    return model_name


def validate_query(ctx, param, query: str):
    if query == "":
        raise click.BadParameter("Query cannot be empty")
    return query


def validate_metadata_options(ctx, param, value):
    metadata_type = ctx.params.get('metadata_type')

    if metadata_type == 'synth':
        if param.name == 'add_domain_knowledge' and value:
            raise click.BadParameter("Domain knowledge cannot be added to synthetic metadata")
        if param.name == 'add_workflow_tags' and value:
            raise click.BadParameter("Workflow tags cannot be added to synthetic metadata")

    return value


def get_agent(
        agent_type: str,
        model_name: str,
        api_config: str,
        log_file: str
):
    if agent_type == 'coder':
        return BaseAgent(
            model_name=model_name,
            api_config=api_config,
            log_file=log_file
        )
    elif agent_type == 'react':
        return ReactAgent(
            model_name=model_name,
            api_config=api_config,
            log_file=log_file
        )
    else:
        raise ValueError(f"Agent {agent_type} not found")


@click.command()
@click.option('--agent_type', type=click.Choice(['coder', 'react']), default='coder', help='Agent type to use for discovery, default is coder')
@click.option('--model_name', callback=validate_model_name, default='gpt-4o', help='Model name, default is gpt-4o, available models are [gpt-4-turbo|llama-3-70b-chat|claude-3-opus|gemini-pro]. Exhaustive list can be found in config/model_config.json')
@click.option('--api_config', default='config/api_config.json', help='API config file, default is config/api_config.json')
@click.option('--log_file', default='discovery_agent.log', help='Log file')
@click.option('--metadata_path', help='Metadata file path', required=True)
@click.option('--metadata_type', type=click.Choice(['real', 'synth']), help='Metadata type', required=True)
@click.option('--add_domain_knowledge', is_flag=True, callback=validate_metadata_options, help='Add domain knowledge to query')
@click.option('--add_workflow_tags', is_flag=True, callback=validate_metadata_options, help='Add Workflow Tags to query')
@click.argument('query', callback=validate_query)
def discovery_agent(
    query: str,
    api_config: str,
    agent_type: str,
    model_name: str,
    log_file: str,
    metadata_path: str,
    metadata_type: str,
    add_domain_knowledge: bool,
    add_workflow_tags: bool
):

    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    metadata_dir = os.path.dirname(metadata_path)
    dataset_paths = [
        f"{metadata_dir}/{metadata['datasets'][i]['name']}"
        for i in range(len(metadata['datasets']))
    ]

    agent = get_agent(
        agent_type=agent_type,
        model_name=model_name,
        api_config=api_config,
        log_file=log_file
    )

    run_autonomous_single_agent_discoverybench(
        agent=agent,
        datasets=dataset_paths,
        metadata=metadata,
        nl_query=query,
        provide_domain_knowledge=add_domain_knowledge,
        provide_workflow_tags=add_workflow_tags,
        dataset_type=metadata_type
    )


if __name__ == '__main__':
    discovery_agent()
