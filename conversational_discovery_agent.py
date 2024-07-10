import traceback
import polars as pl
from dataclasses import dataclass
from typing import Callable
import ipdb
import loguru
import re
from dotenv import load_dotenv
import sys
import pdb
from openai import OpenAI
from flowmason import conduct, load_artifact_with_step_name
from collections import OrderedDict
from typing import List, Dict
import click
from io import StringIO
import json
import os

from packages.constants import SCRATCH_DIR


@dataclass
class CodeRequestPrompt:
    prompt: str

@dataclass
class SynthResultPrompt:
    prompt_template: Callable[[str], str]
    is_code_req: bool

load_dotenv()
api_key = os.environ['THE_KEY']
client = OpenAI(api_key=api_key)  # TODO: put this in an env instead.
logger = loguru.logger

def dag_system_prompt():
    system_prompt = "You are a discovery agent who can execute a python code only once to answer a query based on one or more datasets. The datasets will be present in the current directory. Please write your code in the form of a `flowmason` directed acyclic graph: Here's an example of a flowmason graph:"
    floma_example = """
    ```
    from flowmason import SingletonStep, conduct, load_artifact_with_step_name  # make sure to import flowmason definitions
    from collections import OrderedDict
    CACHE_DIR="scratch/flowmason_cache"
    LOG_DIR="discoveryagent_logs"

    def _step_toy_fn(arg1: float, **kwargs): # all flowmason step functions must have **kwargs for supplying metadata
        print(arg1)
        return 3.1 + arg1

    step_dict = OrderedDict()
    # NOTE: for iterable arguments, use a (hashable) tuple rather than a list
    step_dict['step_singleton'] = SingletonStep(_step_toy_fn, {
        'version': "001", # don't forget to supply version
        'arg1': 2.9
    })
    step_dict['step_singleton_two'] = SingletonStep(_step_toy_fn, {
        'version': "001", 
        'arg1': 'step_singleton'
    })
    run_metadata = conduct(CACHE_DIR, step_dict, LOG_DIR)
    output_step_singleton_two = load_artifact_with_step_name(run_metadata, 'step_singleton_two')
    print(output_step_singleton_two) # 9.1
    ``` 

    Answer the questions in the format:

    Observation: {Full analysis plan in natural language}.
    Code: 
    ```python
    {All code to act on observation.}
    ```
    """
    return system_prompt + floma_example

def loose_system_prompt():
    system_prompt = "You are a discovery agent who can execute a python code only once to answer a query based on one or more datasets. The datasets will be present in the current directory." 
    format_prompt = """Answer the question in the format:
    
    Observation: {Full analysis plan in natural language}.
    Code: 
    ```python
    {All code to act on observation, including any print statements to observe the output.}
    ```
    """
    return f"{system_prompt}\n" + format_prompt

def retrieve_system_prompt(structure_type):
    if structure_type == 'dag':
        return dag_system_prompt()
    elif structure_type == 'loose':
        return loose_system_prompt()
    else:
        raise ValueError("unimplemented system prompt")

def get_column_descriptions(metadata_path):
    path = os.path.dirname(metadata_path)
    with open(metadata_path, 'r') as file:
        dataset = json.load(file)
        columns = eval(f"{dataset['datasets'][0]['columns']['raw']}") # List[Dict] contains 'name' and 'description' keys.
    column_explanations = "\n".join([f"{column['name']}: {column['description']}" for column in columns])
    return column_explanations

def get_csv_path(metadata_path):
    path = os.path.dirname(metadata_path)
    with open(metadata_path, 'r') as file:
        dataset = json.load(file)
    datapath = f"{path}/{dataset['datasets'][0]['name']}" # path to the csv"
    return datapath

def construct_first_message(dataset_path, 
                            query, 
                            structure_type: str,
                            past_traces=None, 
                            explore_first = False):
    # TODO load the columns dataset_description = f"Consider the dataset at {dataset_path}. It contains the following columns: {columns}."
    # get the path containing the dataset
    column_explanations = get_column_descriptions(dataset_path)
    csv_path = get_csv_path(dataset_path)
    query_expanded = f"'{query}'"
    query_expanded = query_expanded + f" Here are the explanations of the columns in the dataset:\n\n{column_explanations}\n"
    if structure_type == 'dag':
        if not past_traces:
            return f"Suppose you had a dataset saved at {csv_path} and you wanted to answer the following query: {query_expanded}\nHow would you go about performing this analysis using a flowmason graph? In the final answer, please write down a scientific hypothesis in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any). Include implementations of any statistical significance tests (if applicable)."
        else:
            first_part = f"Suppose you had a dataset saved at {csv_path}. Here are the explanations of the columns in the dataset:\n\n {column_explanations}\n "
            second_part_traces = "Here are some past queries and associated floma experiments you ran for this dataset (which you can re-use parts of): \n\n" + "\n".join([f"Query: {trace['query']}\nFloma DAG: {trace['floma_dag']}\nFloma Steps: {trace['floma_steps']}" for trace in past_traces]) + "\n\n"
            third_part = f"Now, consider the query: '{query}'. How would you go about performing this analysis using a flowmason graph? In the final answer, please write down a scientific hypothesis in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables. Include implementations of any statistical significance tests (if applicable)."
            complete = first_part + second_part_traces + third_part
            return complete
    elif structure_type == 'loose':
        if past_traces:
            raise ValueError("Past traces are not supported for loose structure type.")
        return f"Suppose you had a dataset saved at {csv_path} and you wanted to answer the following query: {query_expanded}\n Write code to answer the query? In the final answer, please write down a scientific hypothesis in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any). Include implementations of any statistical significance tests (if applicable)."
    else:
        raise ValueError(f"Unimplemented for the structure type: {structure_type}")

# TODO: fill this in.
def generate_exploration_prompt(metadata_path, query):
    column_explanations = get_column_descriptions(metadata_path)
    csv_path = get_csv_path(metadata_path)
    query_expanded = f"'{query}'"
    query_expanded = query_expanded + f" Here are the explanations of the columns in the dataset:\n\n{column_explanations}\n"
    init_message = f"Suppose you had a dataset saved at {csv_path}" +\
        f" and you wanted to answer the following query: {query_expanded}\n"
    init_message = init_message + "Explore the distributions of the relevant columns in the dataset." +\
          " For continuous variables, consider the 5 number summary (median, 1st quartile, 3rd quartile, min, max)." +\
          " For categorical variables, consider the frequency of each category (both absolute and relative; pareto distribution)." +\
          " Write some code to perform this exploration."
    return init_message

def remove_print_statements(code):
    # TODO: if there are indented print statements, that code will still be executed. We should address this later
    return '\n'.join([line for line in code.split('\n') if not line.startswith('print')]) 

def extract_code(response):
    # the code should be encapsulated with ```python and ```.
    # extract the code from the response and return it as a multiline string.
    # find the first occurence of ```python
    response = response[response.index("```"): ]
    if response.startswith("```python"):
        start = response.find("```python") + len("```python")
    elif response.startswith("``` python"):
        start = response.find("``` python") + len("``` python")
    elif response.startswith("```"):
        start = response.find("```") + len("```")
    else:
        ipdb.set_trace()
    # find the first occurence of ```
    end = response.find("```", start)
    # return the code
    return response[start:end]

# def parse_floma_step_impls(code_block: str) -> List[str]:
#     """
#     code_block: a string containing the code block to parse.
#     """
#     # the regex pattern is a function definition that has a **kwargs in the signature and ends with a double newline
#     pattern = r"def\s+_(\w+)\(.*\*\*kwargs\):.*\n.*\n"
#     defns = re.findall(pattern, code_block)
#     return defns

def extract_dataset_name(datapath: str):
    # the dataset name is the part of the path just before the basename
    return os.path.basename(os.path.dirname(datapath))

def parse_floma_step_impls(code_block: str) -> List[str]:
    """
    Parse a code block to extract function definitions that have **kwargs in the signature.

    Parameters:
    code_block (str): A string containing the code block to parse.

    Returns:
    List[str]: A list of strings, each containing a full function definition.

    """
    lines = code_block.split('\n')
    functions = []
    capturing = False
    current_function = []

    for line in lines:
        if capturing:
            if not line.startswith(' ') and not line.startswith('\t'):
                functions.append('\n'.join(current_function))
                current_function = []
                capturing = False
            else:
                current_function.append(line)
        if not capturing and re.match(r'^def\s+_\w+\(.*\*\*kwargs.*\):', line):
            capturing = True
            current_function.append(line)
    
    # In case the last function goes until the end of the code block without returning to indentation level 0
    if current_function:
        functions.append('\n'.join(current_function))
    return functions

def parse_floma_dag(code_block) -> str: # extract the OrderedDict and return it as a string
    # find the index of the line where an OrderedDict is defined. It wont' necessarily be called step_dict
    start = code_block.find("OrderedDict()")
    line_num = code_block.count("\n", 0, start)
    # count the number of characters in the code block up to line_num
    pre_dag_defn = "\n".join(code_block.split("\n")[:line_num])
    start = len(pre_dag_defn)

    # get the variable name of that OrderedDict
    var_name = code_block.split("\n")[line_num].split("=")[0].strip()

    # find the final assignment to the OrderedDict, represented by the variable name.
    last_line_num = 0
    curr_line = 0
    for line in code_block.split("\n"):
        if var_name in line and '=' in line:
            last_line_num = curr_line
        curr_line += 1
            
    final_assignment_position = len("\n".join(code_block.split("\n")[:last_line_num]))
    # go to the end of that final assignment
    end = code_block.find("})", final_assignment_position)
    end = final_assignment_position
    dag = code_block[start:end+1]
    # remove any prefix newlines
    dag = dag.lstrip()
    # for any assignment statement lines, remove the leading whitespace
    dag = '\n'.join([
        line.lstrip() if ('=' in line or '})' in line) else line for line in dag.split('\n')
    ])
    # remove any standalone newlines
    dag = dag.replace("\n\n", "\n").rstrip()
    return dag

def _store_successful_floma_dag(dataset_name: str, query: str, first_response: str):
    floma_dag = parse_floma_dag(first_response) # str
    floma_steps = parse_floma_step_impls(first_response) # List[str]
    path = f"{SCRATCH_DIR}/{dataset_name}_db_traces"
    if not os.path.exists(path):
        os.makedirs(path)
    # check if the query_dags.json file exists.
    # if it does, then read it into memory and append the new floma_dag to it.
    # if it doesn't, then create it and write the floma_dag to it.
    # the file should be a json file with the following structure:
    # [
        #{
#           "query": query,
#           "floma_dag": floma_dag,
#           "floma_steps": floma_steps
        #} 
    # ]
    if os.path.exists(f"{path}/query_dags.json"):
        with open(f"{path}/query_dags.json", 'r') as file:
            data = json.load(file)
        data.append({
            "query": query,
            "floma_dag": floma_dag,
            "floma_steps": floma_steps
        })
        with open(f"{path}/query_dags.json", 'w') as file:
            json.dump(data, file)
    else:
        with open(f"{path}/query_dags.json", 'w') as file:
            json.dump([{
                "query": query,
                "floma_dag": floma_dag,
                "floma_steps": floma_steps
            }], file)
    logger.info(f"Successfully stored the floma_dag for the query: {query} for the dataset: {dataset_name}")

def retrieve_past_dags(dataset_name: str) -> List:
    path = f"{SCRATCH_DIR}/{dataset_name}_db_traces"
    if os.path.exists(f"{path}/query_dags.json"):
        with open(f"{path}/query_dags.json") as f:
            past_traces = json.load(f)
        past_queries = [trace['query'] for trace in past_traces]
        logger.info(f"Retrieved past queries for dataset: {dataset_name}. The past queries are: {past_queries}")
        return past_traces
    else:
        return []

def prompt_for_code_w_error_history(
        prior_implementations: List[str],
        prior_tracebacks: List[str],
        user_code_prompt: str,
):
    assert len(prior_implementations) == len(prior_tracebacks)
    if len(prior_implementations) == 0:
        message = user_code_prompt
    else:
        message = f"{user_code_prompt}.\n\n" +\
            "You provided following code implementations were provided in the past:\n" +\
            "====================\n".join(prior_implementations) +\
            f"\n\n But the following errors were encountered for the prior implementation(s) :\n" +\
            "====================\n".join(prior_tracebacks) +\
            "\n\n Please provide a new and correct code implementation."
    return message


# TODO: implement retries for the code.
def process_code_step(history: List[Dict[str, str]], 
                            user_code_prompt, 
                            current_code: str = "",
                            num_retries = 3) -> str:
    messages = history.copy()

    successful_execution = False
    attempt_num = 0
    prior_implementations = []
    tracebacks = []

    while (not successful_execution) and (attempt_num < num_retries):
        messages.append({"role": "user", "content": prompt_for_code_w_error_history(prior_implementations, 
                                                                                    tracebacks, 
                                                                                    user_code_prompt)})
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2000
        )
        response = response.choices[0].message.content
        extracted_code = extract_code(response)
        stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            complete_code = remove_print_statements(current_code) + "\n" + extracted_code
            exec(complete_code, locals(), locals())
            exec_output = sys.stdout.getvalue()
            successful_execution = True
        except Exception as e:
            sys.stdout = stdout
            # TODO: is there a way to get a stack trace? 
            tb = traceback.format_exc()
            logger.info(f"An error occurred for attempt {attempt_num}: {tb}.")
            logger.info(f"extracted code for attempt {attempt_num}: {extracted_code}")
            tracebacks.append(tb)
            prior_implementations.append(extracted_code)
            messages = messages[:-1] # remove the last message, since the code was incorrect.
            # TODO: need to check that the model can actually repair the code.
            attempt_num += 1

        finally:
            if sys.stdout != stdout:
                sys.stdout = stdout
    if not successful_execution:
        raise Exception(f"Failed to execute the code after {num_retries} attempts.")
    return {'agent_response': response, 'execution_output': exec_output, 'num_attempts': attempt_num}

def process_regular_prompt(history: List[Dict[str, str]],
                           prompt: str) -> str:
    messages = history.copy()
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=2000
    )
    return response.choices[0].message.content

# TODO: need to modify this to return the right structure, so we can probe it.
def process_workflow(system_prompt: str, 
                     workflow: List): 
    messages = [{"role": "system", "content": system_prompt}]
    execution_results = []
    workflow_results = []
    all_code_snippets = [] 
    for i in range(len(workflow)):
        workflow_step = workflow[i]
        if isinstance(workflow_step, CodeRequestPrompt):
            result_dict = process_code_step(messages, workflow_step.prompt, "\n\n".join(all_code_snippets), num_retries=3)
            extracted_code = extract_code(result_dict['agent_response']) # NOTE: Hopefully we don't get duplicate code snippets.
            messages.extend([
                {"role": "user", "content": workflow_step.prompt},
                {"role": "assistant", "content": result_dict['agent_response']}
            ])
            all_code_snippets.append(extracted_code)
            execution_results.append(result_dict['execution_output'])
            workflow_results.append(
                {
                    "prompt": workflow_step.prompt,
                    "response": result_dict['agent_response'],
                    "execution_output": result_dict['execution_output']
                 }
            )
        elif isinstance(workflow_step, SynthResultPrompt):
            prompt = workflow_step.prompt_template(execution_results[-1])
            if workflow_step.is_code_req:
                result_dict = process_code_step(messages, prompt, "\n\n".join(all_code_snippets), num_retries=3)
                messages.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": result_dict['agent_response']}
                ]) # not quite correct, because the next step(s) might also be a code execution. We need to specify this.
                extracted_code = extract_code(result_dict['agent_response'])
                all_code_snippets.append(
                    extracted_code
                )
                workflow_results.append(
                    {
                        "prompt": prompt,
                        "response": result_dict['agent_response'],
                        "execution_output": result_dict['execution_output'], 
                    }
                )
                execution_results.append(result_dict["execution_output"])
            else:
                response = process_regular_prompt(messages, prompt)
                messages.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ])
                workflow_results.append(
                    {"prompt": prompt,
                    "response": response}
                )
        else:
            raise ValueError("Unimplemented workflow step.")
    return workflow_results
            # some of the steps may also be iterable (self-consistency).
            # some of the steps may also have to be repeated (in case of errors.)

def display_results_basic_workflow(workflow, workflow_results):
    # TODO fill in
    pass

def display_results_explore_workflow(workflow, workflow_results):
    for i in range(len(workflow)):
        print(f"======Revolution {i}======")
        user_prompt = workflow_results[i]['prompt']
        print(f"User prompt: {user_prompt}")
        agent_response = workflow_results[i]['response']
        print(f"Agent response: {agent_response}")
        if isinstance(workflow[i], CodeRequestPrompt):
            agent_code = extract_code(agent_response)
            execution_output = workflow_results[i]['execution_output']
            print(f"Agent code: {agent_code}")
            print(f"Execution output: {execution_output}")

def clear_dir(path: str):
    for fname in os.listdir(path):
        os.remove(f"{path}/{fname}")

@click.command()
@click.argument('dataset_path')
@click.argument('query')
@click.option('--save_response', is_flag=True, help='Save the response to a file')
@click.option('--structure-type',
              type=click.Choice(['loose', 'functions', 'dag'], case_sensitive=False))
@click.option('--add_exploration_step', is_flag=True, help='Have the agent first explore the univariate distributions of the dataset.')
@click.option('--generate_visual',is_flag=True) 
def analyze_dataset(dataset_path: str, query: str, 
                    save_response: bool, 
                    structure_type: str, 
                    generate_visual: bool, 
                    add_exploration_step: bool):
    # the workflow is a list of the form [str, Function[str] -> str, Function[str] -> str, ...]
    past_traces = retrieve_past_dags(extract_dataset_name(dataset_path))
    # identify the workflow to use:
    if past_traces and not add_exploration_step:
        first_message = construct_first_message(dataset_path, query, structure_type, past_traces)
    elif add_exploration_step:
        exploration_step = generate_exploration_prompt(dataset_path, query)
        exploration_passback = lambda exploration_result: f"The output of the previous code is:\n{exploration_result}\n\n" +\
            f"Now, how would you go about performing an analysis of the query '{query}' using a flowmason graph?" +\
            " In the final answer, please write down a scientific hypothesis in natural language" +\
            " derived from the provided dataset, clearly stating the context of hypothesis (if any)" +\
            " , variables chosen (if any) and relationship between those variables (if any). Include implementations" +\
            " of any statistical significance tests (if applicable)." 
        rev_second_prompt_template = lambda exec_result: f"The output of the previous code is:" +\
            f"\n{exec_result}\n\nBased on this analysis output, what is your scientific hypothesis for the query '{query}'?" +\
            f"Ground your answer in the context of the analysis (including any metrics or statistical significance tests)." +\
            f"Keep your response succinct and clear."
        workflow = [CodeRequestPrompt(exploration_step), SynthResultPrompt(exploration_passback, True), 
                    SynthResultPrompt(rev_second_prompt_template, False)]
        execution_results = process_workflow(retrieve_system_prompt(structure_type), workflow)
        display_results_explore_workflow(workflow, execution_results)
        # clear the floma cache at scratch/flowmason_cache
        clear_dir("scratch/flowmason_cache")
    else:
        first_message = construct_first_message(dataset_path, query, structure_type)
        rev_second_prompt_template = lambda exec_result: f"The output of the previous code is:\n{exec_result}\n\nBased on this output, what is your scientific hypothesis?"
        workflow = [CodeRequestPrompt(first_message), SynthResultPrompt(rev_second_prompt_template, False)] 
        execution_results = process_workflow(retrieve_system_prompt(structure_type), workflow)
        display_results_basic_workflow(workflow, execution_results)
        ipdb.set_trace()

    # # hypothesis = second_response.choices[0].message.content
    # print("=====Agent observation + code=====")
    # print(code_res_dict['rev_agent_first_response'])
    # print("=====Agent execution output=====")
    # print(code_res_dict['rev_execution_res'])
    # print("=====Agent hypothesis=====")
    # print(code_res_dict['rev_agent_second_response'])

    # TODO: write the floma dag + steps to a json file.
    dataset_name = extract_dataset_name(dataset_path)
    if save_response:
        _store_successful_floma_dag(dataset_name, query, extract_code(code_res_dict['rev_execution_res']))
    
    if generate_visual:
        if structure_type == 'dag':
            generate_visual_message = f"Visualize the data with respect to your prior hypothesis. Do this by appending a visualization step (or multiple steps) to the flowmason graphs. Save the visual to 'visual.png'. Generate a caption for the visual"
        elif structure_type == 'loose':
            generate_visual_message = f"Visualize the data with respect to your prior hypothesis. Do this by writing code to visualize the data. Save the visual to 'visual.png'. Generate a caption for the visual."
        # query the agent again to visualize the data with repsect to the hypothesis
        messages = [
                {"role": "system", "content": full_system_prompt},
                {'role': 'user', "content": first_message},
                {'role': 'assistant', "content": code_res_dict['rev_first_response']},
                {'role': 'user', "content": rev_second_prompt_template(code_res_dict['rev_execution_res'])},
                {'role': 'assistant', "content": code_res_dict['rev_second_response']}
        ]
        visual_output = process_code_revolution(messages, generate_visual_message)
        print("=====Agent visualization output=====")
        print(visual_output)

if __name__ == '__main__':
    analyze_dataset()