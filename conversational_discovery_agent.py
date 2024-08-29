import json
import numpy as np
import traceback
import polars as pl
from functools import partial
import openai
import pandas as pd
from typing import Union, Tuple, Iterable
from itertools import product
from dataclasses import dataclass
import random
from typing import Callable
import ipdb
import loguru
import re
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import sys
import pdb
from openai import OpenAI
from flowmason import conduct, load_artifact_with_step_name, SingletonStep, MapReduceStep
from collections import OrderedDict, Counter 
from typing import List, Dict, Optional
import click
from io import StringIO
import json
import os
import dill

from packages.library_learn import create_ll_prompt, step_write_leap_helper,\
    step_extract_helper, step_prettyprint_helper, step_write_standard_abstraction,\
    step_describe_incorrect_programs, step_collect_abstractions, step_collect_programs,\
    format_program_descriptions, parse_library_selection_response
from packages.constants import SCRATCH_DIR, MY_CACHE_DIR, MY_LOG_DIR, LEAP_CACHE_DIR, DB_ANALYSIS_FAILED, EMPTY_SC_ANSWER, INDIVIDUAL_QUERY_DIR,\
    INDIVIDUAL_Q_LOGS
from packages.db_data_format import load_db_dataframe, generate_variable_exploration_prompt
from packages.agent_dataclasses import ExperimentResult, FailedExperimentResult, CodeRequestPrompt,\
    SynthResultPrompt, CodeExtractionException, CodeRetryException, LongPromptException,\
    SelfConsistencyExperimentResult, FailedSelfConsistencyExperimentResult, ContrastiveProgramExample,\
    Program, DBDataset
from packages.json_repair import repair_json
from packages.answer_extract_utils import extract_db_label, extract_db_rationale


load_dotenv()
api_key = os.environ['THE_KEY']
client = OpenAI(api_key=api_key)  # TODO: put this in an env instead.
logger = loguru.logger


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

def retrieve_system_prompt(structure_type, floma_cache="flowmason_cache"):
    if structure_type == 'loose':
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

def get_column_description_dict(metadata_path) -> Dict[str, Iterable[Tuple[str, str]]]:
    dataset_to_column_descriptions = []
    with open(metadata_path, 'r') as file:
        meta_dataset = json.load(file)
        for dataset in meta_dataset['datasets']:
            columns = eval(f"{dataset['datasets'][0]['columns']['raw']}") # List[Dict] contains 'name' and 'description' keys.
            column_explanations = [(column['name'], column['description']) for column in columns]
            dataset_to_column_descriptions[dataset['name']] = column_explanations
    return dataset_to_column_descriptions 

def get_csv_paths(metadata_path) -> List[str]:
    path = os.path.dirname(metadata_path)
    with open(metadata_path, 'r') as file:
        dataset = json.load(file)
    datapaths = [f"{path}/{dataset['datasets'][i]['name']}" for i in range(len(dataset['datasets']))]
    return datapaths

def construct_first_message(dataset_path, 
                            query, 
                            structure_type: str,
                            past_traces=None, 
                            explore_first = False):
    # TODO load the columns dataset_description = f"Consider the dataset at {dataset_path}. It contains the following columns: {columns}."
    # get the path containing the dataset
    column_explanations = get_column_descriptions(dataset_path) # Dict[str, Iterable[Tuple[str, str]]]
    csv_path = get_csv_paths(dataset_path) # List[str]
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

def generate_exploration_prompt(metadata_path, query):
    column_explanations = get_column_descriptions(metadata_path)
    csv_path = get_csv_path(metadata_path)
    query_expanded = f"'{query}'"
    filetype = csv_path.split('.')[-1]
    if filetype == "dta":
        dtypes = pd.read_stata(csv_path).dtypes.to_dict()
        num_datapoints = len(pd.read_stata(csv_path))
    elif filetype == "csv":
        dtypes = pd.read_csv(csv_path).dtypes.to_dict()
        num_datapoints = len(pd.read_csv(csv_path))
    elif filetype == "tsv":
        dtypes = pd.read_csv(csv_path, sep='\t').dtypes.to_dict()
        num_datapoints = len(pd.read_csv(csv_path, sep='\t'))
    query_expanded = query_expanded + f" Here are the explanations of the columns in the dataset:\n\n{column_explanations}\n"
    init_message = f"Suppose you had a dataset saved at {csv_path}" +\
        f" and you wanted to answer the following query: {query_expanded}\n"
    init_message = init_message + "Explore the distributions of the relevant columns in the dataset." +\
          " The data types of the columns are:\n" + str(dtypes) + ".\n" +\
          f" There are {num_datapoints} datapoints in the dataset." +\
          " For continuous variables, consider the 5 number summary (median, 1st quartile, 3rd quartile, min, max)." +\
          " For categorical variables, consider the frequency of each category (both absolute and relative; pareto distribution)." +\
          " Make sure to obtain summary statistics or view a few random samples (or both), rather than printing all of the data in a column (which may flood the console)." +\
          " Write some code to perform this exploration. Do not try to answer the question yet. Don't draw any plots and don't use df.info."
    return init_message

            

def qrdata_generate_exploration_prompt(qrdata_block: Dict):
    # the qrdata block is a dictionary containing the following:
    # data_description, question, data_files (List), question_type (multiple_choice/numerical)
    # when the question type is multiple choice, there is also a multiple_choices key (List), enumerating
    # the options.
    data_description = qrdata_block['data_description']
    question = qrdata_block['question']
    data_files = qrdata_block['data_files'] # usually a list with one element, but sometimes more.
    question_type = qrdata_block['meta_data']['question_type']
    # if question_type == 'multiple_choice':
    #     possible_answers = f" (possible answers): {str(qrdata_block['meta_data']['multiple_choices'])}"
    # else:
    #     possible_answers = ""

    dataset_prefix = "data/qrdata/"
    # data_paths = "\n".join(data_files)
    # data_paths = "\n".join([f"{dataset_prefix}{data_file}" for data_file in data_files])
    data_paths = [f"{dataset_prefix}{data_file}" for data_file in data_files]
    all_columns = []
    for data_path in data_paths:
        all_columns.append(pd.read_csv(data_path).columns.to_list())
    dtypes_str = ""
    for i in range(len(data_files)):
        dtypes = pd.read_csv(data_paths[i]).dtypes.to_dict()
        dtypes_str += f"Dataset {i+1}: {str(dtypes)}\n"
    determiner = "a" if len(data_files) == 1 else f"{len(data_files)}"
    plural_aux = "is" if len(data_files) == 1 else "are"
    plural_nom = "s" if len(data_files) == 1 else ""
    init_message = f"Suppose you had {determiner} dataset{plural_nom} saved at {data_paths}." +\
        f" The dataset{plural_nom} {plural_aux} described as follows: {data_description}" +\
        f" The columns in the dataset{plural_nom} are: {all_columns}." +\
        f" You are asked the following question: {question}.\n\n" +\
        f" Perform an initial exploration of the distributions of the relevant columns in the dataset{plural_nom}." +\
        f" The data types of the columns are:\n{dtypes}." +\
        " For continuous variables, consider the 5 number summary (median, 1st quartile, 3rd quartile, min, max)." +\
        " For categorical variables, consider the frequency of each category (both absolute and relative; pareto distribution)." +\
        " Write some code to perform this initial exploration. Do not try to answer the question yet. Don't draw any plots though, and don't use df.info."
    return init_message

def generate_dataset_description_and_query_only_prompt(metadatapath, query):
    column_explanations = get_column_descriptions(metadatapath)
    csv_path = get_csv_path(metadatapath)
    query_expanded = f"'{query}'"
    query_expanded = query_expanded + f" Here are the explanations of the columns in the dataset:\n\n{column_explanations}\n"
    init_message = f"Suppose you had a dataset saved at {csv_path}" +\
        f" and you wanted to answer the following query: {query_expanded}\n"
    return init_message

def remove_print_statements(code):
    # TODO: if there are indented print statements, that code will still be executed. We should address this later
    return '\n'.join([line for line in code.split('\n') if not line.startswith('print')]) 

def obtain_num_errors(workflow_results):
    # use the 'num_attempts' key in the workflow_results to obtain the number of errors.
    # if that key is present
    return sum([result['num_attempts'] for result in workflow_results if 'num_attempts' in result])

def extract_code(response):
    # the code should be encapsulated with ```python and ```.
    # extract the code from the response and return it as a multiline string.
    # find the first occurence of ```python
    try:
        response = response[response.index("```"): ]
    except ValueError:
        logger.error(f"``` backticks not found")
        raise CodeExtractionException("``` backticks not found", "n/a", "n/a", response)
    if response.startswith("```python"):
        start = response.find("```python") + len("```python")
    elif response.startswith("``` python"):
        start = response.find("``` python") + len("``` python")
    elif response.startswith("```"):
        start = response.find("```") + len("```")
    else:
        raise CodeExtractionException("The code block does not start with ```python or ```", "n/a")
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
            "\n\n Please provide a new and correct code implementation." + \
            "Remember to update the version number of the function if you update its implementation." +\
            "If there is a ModuleNotFoundError on an import statement, then it is mispelled or it is not installed (more likely). If it is not installed, you should not try to use it in the code." 
    return message


# TODO: implement retries for the code.
def process_code_step(history: List[Dict[str, str]], 
                            user_code_prompt, 
                            workflow_i: int,
                            current_code: str = "",
                            num_retries = 3, 
                            oai_model: str = "gpt-4") -> str:
    messages = history.copy()

    successful_execution = False
    attempt_num = 0
    prior_implementations = []
    tracebacks = []

    total_prompt_tokens = 0
    total_response_tokens = 0
    while (not successful_execution) and (attempt_num < num_retries):
        messages.append({"role": "user", "content": prompt_for_code_w_error_history(prior_implementations, 
                                                                                    tracebacks, 
                                                                                    user_code_prompt)})
        try:
            response = client.chat.completions.create(
                model=oai_model,
                messages=messages,
                max_tokens=3000
            )
        except openai.BadRequestError as e:
            assert "Please reduce the length" in e.message or "string too long" in e.message, ipdb.set_trace()
            raise e
        except openai.RateLimitError as e:
            raise e
        total_prompt_tokens += response.usage.prompt_tokens
        total_response_tokens += response.usage.completion_tokens
        response = response.choices[0].message.content
        try:
            extracted_code = extract_code(response)
        except CodeExtractionException:
            raise CodeExtractionException(f"Code could not be extracted from the response:\n\n{response} .", -1, "n/a", response)
        stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            complete_code = remove_print_statements(current_code) + "\n" + extracted_code
            exec(complete_code, locals(), locals())
            exec_output = sys.stdout.getvalue() # TODO: there might be warnings that are predicted to stderr, which we should also try to capture.
            successful_execution = True
        except Exception as e:
            sys.stdout = stdout
            # TODO: is there a way to get a stack trace? 
            try:
                tb = traceback.format_exc()
            except KeyError as ke:
                tb = f"Exception during formatting traceback; complete traceback unavailable. Error message is: {e}"
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
        raise CodeRetryException("n/a", workflow_i, f"Failed to execute the code after {num_retries} attempts.", 
                                 prior_implementations, tracebacks)
    return {'agent_response': response, 
            'execution_output': exec_output, 
            'num_attempts': attempt_num, 
            'total_prompt_tokens': total_prompt_tokens,
            'total_response_tokens': total_response_tokens}

def process_regular_prompt(history: List[Dict[str, str]],
                           prompt: str) -> Dict:
    messages = history.copy()
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000
    )
    num_prompt_tokens = response.usage.prompt_tokens
    num_response_tokens = response.usage.completion_tokens
    # return response.choices[0].message.content
    return {'agent_response': response.choices[0].message.content,
            'num_prompt_tokens': num_prompt_tokens,
            'num_response_tokens': num_response_tokens}

def process_workflow(system_prompt: str, 
                     workflow: List, 
                     oai_model: str = "gpt-4") -> List[Dict]: 
    messages = [{"role": "system", "content": system_prompt}]
    execution_results = []
    workflow_results = []
    all_code_snippets = [] 


    # if library_programs:
    #     all_code_snippets.append(
    #         extract_code(library_programs)
    #     )
    for i in range(len(workflow)):
        workflow_step = workflow[i]
        if isinstance(workflow_step, CodeRequestPrompt):
            try:
                result_dict = process_code_step(messages, workflow_step.prompt, i, "\n\n".join(all_code_snippets), num_retries=3, oai_model=oai_model)
            except openai.BadRequestError as e:
                raise LongPromptException(workflow_results, i, e.message, [], [])
            except openai.RateLimitError as e:
                raise LongPromptException(workflow_results, i, e.message, [], [])
            except CodeRetryException as e:
                raise CodeRetryException(workflow_results, i, e.message, e.prior_implementations, e.prior_tracebacks)
            except CodeExtractionException as e:
                raise CodeExtractionException(e.message, i, workflow_results, e.agent_response)
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
                    "execution_output": result_dict['execution_output'],
                    "num_attempts": result_dict['num_attempts'],
                    "total_prompt_tokens": result_dict['total_prompt_tokens'],
                    "total_response_tokens": result_dict['total_response_tokens']
                 }
            )
        elif isinstance(workflow_step, SynthResultPrompt):
            prompt = workflow_step.prompt_template(execution_results[-1])
            if workflow_step.has_code_provision:
                all_code_snippets.append(
                    extract_code(prompt)
                )
            if workflow_step.is_code_req:
                try:
                    result_dict = process_code_step(messages, prompt, i, "\n\n".join(all_code_snippets), num_retries=3, oai_model=oai_model)
                except openai.BadRequestError as e:
                    raise LongPromptException(workflow_results, i, e.message)
                except openai.RateLimitError as e:
                    raise LongPromptException(workflow_results, i, e.message)
                except CodeRetryException as e:
                    raise CodeRetryException(workflow_results, i, e.message, e.prior_implementations, e.prior_tracebacks)
                except CodeExtractionException as e:
                    raise CodeExtractionException(e.message, i, workflow_results, e.agent_response)
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
                        "num_attempts": result_dict['num_attempts'],
                        "total_prompt_tokens": result_dict['total_prompt_tokens'],
                        "total_response_tokens": result_dict['total_response_tokens']
                    }
                )
                execution_results.append(result_dict["execution_output"])
            else:
                try:
                    response_dict = process_regular_prompt(messages, prompt)
                except openai.BadRequestError as e: 
                    raise LongPromptException(workflow_results, i, e.message)
                except openai.RateLimitError as e:
                    raise LongPromptException(workflow_results, i, e.message)
                messages.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_dict['agent_response']}
                ])
                workflow_results.append(
                    { 
                        "prompt": prompt,
                        "response": response_dict['agent_response'],
                        "total_prompt_tokens": response_dict['num_prompt_tokens'],
                        "total_response_tokens": response_dict['num_response_tokens']
                    }
                )
                execution_results.append(response_dict['agent_response']) # TODO: need to check if this breaks anything.
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

def generate_self_consistency_prompt(final_answers: List[str]) -> str:
    final_answers_str = ""
    # add ====== Answer {i} ====== for each final answer
    # {final_answer i}
    for i, final_answer in enumerate(final_answers):
        final_answers_str += f"====== Answer {i} ======\n{final_answer}\n"
    
    final_answers_str += "\n\nWhich of the above answers is most preferred? Return your answer and your rationale for picking that answer in the form: '({i}) {rationale}'. Use 0 based indexing."
    return final_answers_str

    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {'role': 'user', 'content': f""}

def represent_analysis(workflow_results: List[Dict[str, str]]) -> str:
    # TODO: implement this so the whole experiment code is represented.
    # code_response_one = extract_code(workflow_results[0]['response'])
    code_response_two = extract_code(workflow_results[1]['response'])
    execution_output_two = workflow_results[1]['execution_output']
    final_interpretation = workflow_results[-1]['response']

    analysis_representation = f"=== Analysis code ===\n{code_response_two}\n\n=== Execution output ===\n{execution_output_two}\n\n=== Final interpretation ===\n{final_interpretation}"
    return analysis_representation

def _extract_response_db(response):
    response_kw = "RESPONSE:"
    response_index = response.index(response_kw)

    response = response[response_index + len(response_kw):].strip()
    return response

def build_experiment_result_qrdata(execution_results, answer_dict) -> ExperimentResult:
    total_prompt_tokens = sum([result['total_prompt_tokens'] for result in execution_results])
    total_response_tokens = sum([result['total_response_tokens'] for result in execution_results])
    total_errors = obtain_num_errors(execution_results)
    return ExperimentResult("success", 
                            workflow=execution_results,
                            num_prompt_tokens=total_prompt_tokens,
                            num_response_tokens=total_response_tokens,
                            answer = answer_dict['answer'], 
                            num_errors=total_errors)

def build_experiment_result_db(execution_results, claim) -> ExperimentResult:
    total_prompt_tokens = sum([result['total_prompt_tokens'] for result in execution_results])
    total_response_tokens = sum([result['total_response_tokens'] for result in execution_results])
    total_errors = obtain_num_errors(execution_results)
    return ExperimentResult("success",
                            workflow=execution_results,
                            num_prompt_tokens=total_prompt_tokens,
                            num_response_tokens=total_response_tokens,
                            answer = claim,
                            num_errors=total_errors)

def execute_pass_db(workflow, structure_type, floma_cache, 
                    library_programs: Optional[Iterable[Program]] = None, 
                    oai_model: str = "gpt-4") -> ExperimentResult:
    try:
        execution_results = process_workflow(retrieve_system_prompt(structure_type, floma_cache), workflow, oai_model=oai_model)
        clear_dir(f"scratch/{floma_cache}")
        answer = _extract_response_db(execution_results[-1]['response'])
        return build_experiment_result_db(execution_results, answer)
    except CodeRetryException as e:
        message = e.message
        prior_implementations = e.prior_implementations
        prior_tracebacks = e.prior_tracebacks
        workflow_i = e.workflow_i
        logger.error(f"Error occurred during workflow step {workflow_i}: {message}.\nThe most recent implementation was: {prior_implementations[-1]}.")
        workflow_so_far = e.workflow_so_far
        clear_dir(f"scratch/{floma_cache}")
        return FailedExperimentResult(workflow_i,
                                    prior_tracebacks,
                                    "code_retry_exception",
                                    prior_implementations,
                                    workflow_so_far
                                    )
    except LongPromptException as e:
        logger.error(f"BadRequestError occurred: {e}.")
        clear_dir(f"scratch/{floma_cache}")
        return FailedExperimentResult(e.workflow_i,
                                    [],
                                    "long_prompt_exception",
                                    [],
                                    e.workflow_so_far
                                        )
    except CodeExtractionException as e:
        logger.error(f"Error occurred during code extraction: {e}.")
        clear_dir(f"scratch/{floma_cache}")
        return FailedExperimentResult(e.workflow_i,
                                    [],
                                    "code_extraction_exception",
                                    [],
                                    e.workflow_so_far, 
                                    e.agent_response
                                    )

        # print(f"Error occurred: {e}.")
        # ipdb.set_trace()

    # TODO: implement build_experiment_result and cache the result. 
    # return build_experiment_result(execution_results, answer_dict)


def complete_pass_qrdata(workflow, structure_type):
    try:
        execution_results = process_workflow(retrieve_system_prompt(structure_type), workflow)
        clear_dir("scratch/flowmason_cache")
        answer_dict = _extract_json(execution_results[-1]['response'])
        return build_experiment_result_qrdata(execution_results, answer_dict)
        # if add_exploration_step:
        #     display_results_explore_workflow(workflow, execution_results)
        # elif not add_exploration_step:
        #     display_results_basic_workflow(workflow, execution_results)
    except CodeRetryException as e: 
        message = e.message
        prior_implementations = e.prior_implementations
        prior_tracebacks = e.prior_tracebacks
        workflow_i = e.workflow_i
        logger.error(f"Error occurred during workflow step {workflow_i}: {message}.\nThe most recent implementation was: {prior_implementations[-1]}.")
            # def __init__(self, failed_phase_i, 
            #  tracebacks, 
            #  fail_type: str, 
            #  implementation_attempts: List[str],
            #  workflow_so_far):
        workflow_so_far = e.workflow_so_far
        clear_dir("scratch/flowmason_cache")
        return FailedExperimentResult(workflow_i,
                                    prior_tracebacks,
                                    "code_retry_exception",
                                    prior_implementations,
                                    workflow_so_far
                                    )
        # did it fail on the exploration step or the analysis step?
        # we want to see what it failed on too.
    except LongPromptException as e:
        logger.error(f"BadRequestError occurred: {e}.")
        clear_dir("scratch/flowmason_cache")
        return FailedExperimentResult(e.workflow_i,
                                    [],
                                    "long_prompt_exception",
                                    [],
                                    e.workflow_so_far
                                        )
    except CodeExtractionException as e:
        logger.error(f"Error occurred during JSON extraction: {e}.")
        clear_dir("scratch/flowmason_cache")
        return FailedExperimentResult(2,
                                    [],
                                    "code_extraction_exception",
                                    [],
                                    e.workflow_so_far
                                    )
    except SyntaxError as e:
        logger.error(f"Syntax error occurred: {e}.")
        clear_dir("scratch/flowmason_cache")
        return FailedExperimentResult(-1,
                                    [],
                                    "syntax_error",
                                    [],
                                    execution_results
                                    )
    except ExtractJSONException as e:
        logger.error(f"Error occurred during JSON extraction: {e}.")
        clear_dir("scratch/flowmason_cache")
        return FailedExperimentResult(2,
                                    [],
                                    "json_extraction_error",
                                    [],
                                    execution_results
                                    )

def analyze_qrdata(structure_type: bool, add_exploration_step: bool, 
                   generate_visual: bool, 
                   self_consistency: bool, index: int, 
                   **kwargs):
    # with open("data/qrdata/QRDataClean.json", 'r') as file:
    with open("data/qrdata/QRData.json", 'r') as file:
        qrdata = json.load(file)
    if index == -1:
        random_ind = random.randint(0, len(qrdata)- 1)
    else:
        random_ind = index
    qr_datapoint = qrdata[random_ind]
    query = qr_datapoint['question']
    exploration_step = qrdata_generate_exploration_prompt(qr_datapoint)
    # TODO: for the unstructured version, need to change the exploration passback.
    if structure_type == 'dag':
        exploration_passback = lambda exploration_result: f"The output of the previous code is:\n{exploration_result}\n\n" +\
            f"Now, how would you go about performing an analysis of the query '{query}' using a flowmason graph?"
    elif structure_type == 'loose':
        exploration_passback = lambda exploration_result: f"The output of the previous code is:\n{exploration_result}\n\n" +\
        f"Now, how would you go about performing an analysis of the query '{query}'?"
    else:
        raise ValueError("Unimplemented structure type.")
    analysis_passback = lambda exec_result: f"The output of the previous code is:\n{exec_result}\n\nBased on this output, write the final answer" +\
        " in a JSON dictionary, with two keys: 'answer' and 'rationale'. The 'answer' key should contain the final answer to the query" +\
        " and the 'rationale' key should contain the rationale for the answer (make sure the rationale value doesn't have internal quotes). The answer should be a numeric value or a short phrase for the question. Don't write anything else except for the JSON object."
        # " as a numeric value or a short phrase for the question. Don't write anything else." 
    workflow = [
                CodeRequestPrompt(exploration_step), 
                SynthResultPrompt(exploration_passback, True), 
                SynthResultPrompt(analysis_passback, False)
                ] 
    if not self_consistency:
        return complete_pass_qrdata(workflow, structure_type)
    else:
        SELF_CONSISTENCY_ITERS = 5
        final_analyses = []
        sc_runs = []
        for _ in range(SELF_CONSISTENCY_ITERS):
            logger.info(f"Starting self-consistency iteration {_}.")
            experiment_result = complete_pass_qrdata(workflow, structure_type) # either ExperimentResult or FailedExperimentResult
            if hasattr(experiment_result, 'workflow'): # this is how you know it succeeded.
                # execution_results = process_workflow(retrieve_system_prompt(structure_type), workflow)
                # final_analysis = execution_results[-1]['response']
                analysis_representation = represent_analysis(experiment_result.workflow)
                # answer_dict = _extract_json(execution_results[-1]['response'])
                final_analyses.append(analysis_representation)
            sc_runs.append(experiment_result)
            logger.info(f"Completing self-consistency iteration {_}.")
        assert len(sc_runs) == SELF_CONSISTENCY_ITERS
        assert len(final_analyses) >= 1  and len(final_analyses) <= SELF_CONSISTENCY_ITERS, ipdb.set_trace() # at least one needs to be successful.
        self_consistency_prompt = generate_self_consistency_prompt(exploration_step, final_analyses)
        sc_response = process_regular_prompt([], self_consistency_prompt)
        ipdb.set_trace()
        return SelfConsistencyExperimentResult(tuple(sc_runs), sc_response)

def step_evaluate_resampled_analysis(resampled_result: Union[ExperimentResult, FailedExperimentResult, int], 
                                     query: str, 
                                     gold_hypothesis: str,
                                     eval_all_samples: Optional[bool] = False,
                                     **kwargs) -> str:
    if resampled_result == -1:
        return "Not a resampled result."
    else:
        evaluation = step_evaluate_claim(
            resampled_result,
            query,
            gold_hypothesis,
            eval_all_samples,
        )
        return evaluation

def step_evaluate_claim(experiment_result: Union[ExperimentResult, FailedExperimentResult], 
                        query: str,
                        gold_hypothesis, 
                        eval_all_samples: Optional[bool] = False,
                        **kwargs) -> Union[str, Iterable[str]]: 
    print("At evaluation stage.") 

    if hasattr(experiment_result, 'answer'):
        predicted_answer = experiment_result.answer
        prompt = f"Consider the query: '{query}'. Then, Consider the following claim (C1):\n\n'{gold_hypothesis}'.\n\nThen, consider the analysis A1:\n\n'{predicted_answer}'\n\nDoes A1 support C1 (SUPP), partially support C1 (PRT_SUPP),  refute C1 (REFUTE), or not provide enough info (LACK_INFO)? Write your answer in the form of a json with two keys: label and rationale." 
        match_response = process_regular_prompt([], prompt)
        return match_response
    elif hasattr(experiment_result, 'final_answer'):
        if all([hasattr(x, 'failed_phase_i') for x in experiment_result.individual_runs]): # NOTE: we use hasattr because the isinstance is behaving weirdly.
            ipdb.set_trace()
        
        if not eval_all_samples:
            predicted_answer = _retrieve_sc_run_answer(experiment_result)
            prompt = f"Consider the query: '{query}'. Then, Consider the following claim (C1):\n\n'{gold_hypothesis}'.\n\nThen, consider the analysis A1:\n\n'{predicted_answer}'\n\nDoes A1 support C1 (SUPP), partially support C1 (PRT_SUPP),  refute C1 (REFUTE), or not provide enough info (LACK_INFO)? Write your answer in the form of a json with two keys: label and rationale." 
            match_response = process_regular_prompt([], prompt)
            return match_response
        elif eval_all_samples:
            claims = []
            completed_runs = [x for x in experiment_result.individual_runs] 
            for run in completed_runs:
                if hasattr(run, 'fail_type'):
                    claims.append("The experiment failed, so the claim cannot be evaluated")
                else:
                    predicted_answer = run.answer
                    prompt = f"Consider the query: '{query}'. Then, Consider the following claim (C1):\n\n'{gold_hypothesis}'.\n\nThen, consider the analysis A1:\n\n'{predicted_answer}'\n\nDoes A1 support C1 (SUPP), partially support C1 (PRT_SUPP),  refute C1 (REFUTE), or not provide enough info (LACK_INFO)? Write your answer in the form of a json with two keys: label and rationale." 
                    claims.append(process_regular_prompt([], prompt))
            return tuple(claims)
    elif hasattr(experiment_result, 'fail_type'):
        return "The experiment failed, so the claim cannot be evaluated."
    elif hasattr(experiment_result, 'failed_runs'):
        return "The experiment failed, so the claim cannot be evaluated."

def complete_pass_db(dataset_path, query, 
                     structure_type, generate_visual,
                     add_exploration_step, self_consistency, 
                     nl_workflow: Optional[str] = None, 
                     failure_memories: Optional[Iterable[str]] = None,
                     library_programs: Optional[Iterable[Program]] = None,
                     use_variable_filtered_exploration: Optional[bool] = False,
                     oai_model: Optional[str] = "gpt-4",
                     **kwargs):
    if nl_workflow:
        assert failure_memories is not None, ipdb.set_trace()
        hint_str = (" Consider using the following hint instead:\n\n" + f"{nl_workflow}\n\n") if nl_workflow else ""
        failure_memories_str = " Previous attempts at answering this query failed. Here are the previous attempts:\n" + "\n".join([f"{i}. {memory}" for i, memory in enumerate(failure_memories)]) if failure_memories else ""
    else:
        hint_str = ""
        failure_memories_str = ""

    if not library_programs:
        # TODO: add another step here.
        exploration_passback = lambda exploration_result: f"The output of the previous code is:\n{exploration_result}\n\n" +\
            f"Now, how would you go about performing an analysis of the query '{query}'?" +\
            failure_memories_str +\
            hint_str +\
            " In the final answer, please write down a scientific hypothesis in natural language" +\
            " derived from the provided dataset, clearly stating the context of hypothesis (if any)" +\
            " , variables chosen (if any) and relationship between those variables (if any). Include implementations" +\
            " of any statistical significance tests (if applicable)." +\
            (f"Consider using or adapting any of the following helper functions: {library_programs}" if library_programs else "") +\
            f"to answer the query '{query}"
        rev_second_prompt_template = lambda exec_result: f"The output of the previous code is:" +\
            f"\n{exec_result}\n\nBased on this analysis output, what is your answer for the query '{query}'?" +\
            f"Ground your answer in the context of the analysis (including any metrics and statistical significance tests)." +\
            f"Keep your response short and to the point (1-3 sentences). Start your response with 'RESPONSE:'"
        if use_variable_filtered_exploration:
            exploration_step = generate_variable_exploration_prompt(dataset_path, query)
        else:
            exploration_step = generate_exploration_prompt(dataset_path, query)
        workflow = [
                    CodeRequestPrompt(exploration_step), 
                    SynthResultPrompt(exploration_passback, True), 
                    SynthResultPrompt(rev_second_prompt_template, False)
                    ]
    else:
        # TODO: fill this in.
        # variable_selection_step = generate_variable_selection_prompt(dataset_path, query)
        if use_variable_filtered_exploration:
            exploration_step = generate_variable_exploration_prompt(dataset_path, query)
        else:
            exploration_step = generate_exploration_prompt(dataset_path, query)
        exploration_passback = lambda exploration_result: f"The output of the previous code is:\n{exploration_result}\n\n" +\
            f"Now, considering the query '{query}'," +\
            f"would you use or adapt any of the following helper functions?\n\n{format_program_descriptions(library_programs)}\n\n" +\
            f"Return your response in the form of a JSON object with two keys: 'indices' and 'rationale'." + \
                "The 'indices' key should contain a list of the indices" + \
                "of the helper functions you would use or adapt, and the 'rationale' key should contain a rationale for your choice." 
        analysis_interpretation_template = lambda exec_result: f"The output of the analysis is:" +\
            f"\n{exec_result}\n\nBased on this analysis output, what is your answer for the query '{query}'?" +\
            f"Ground your answer in the context of the analysis (including any metrics and statistical significance tests)." +\
            f"Keep your response short and to the point (1-3 sentences). Start your response with 'RESPONSE:'"
        workflow = [
                    CodeRequestPrompt(exploration_step),
                    SynthResultPrompt(exploration_passback, False),  # request agent to select library function(s)
                    SynthResultPrompt(partial(parse_library_selection_response, query, library_programs), True, True),  # request agent to write the analysis 
                    SynthResultPrompt(analysis_interpretation_template, False) # agent interprets the execution output from the analysis code
                    ]
    # TODO: the library usage is encoded in the workflow, so we don't need it a an argument for execute_pass_db, I think.
    experiment_result = execute_pass_db(workflow, structure_type, "flowma_cache", oai_model=oai_model)
    return experiment_result

# @click.command()
# @click.argument('dataset_path')
# @click.argument('query')
# @click.option('--save_response', is_flag=True, help='Save the response to a file')
# @click.option('--structure-type',
#               type=click.Choice(['loose', 'functions', 'dag'], case_sensitive=False))
# @click.option('--add_exploration_step', is_flag=True, help='Have the agent first explore the univariate distributions of the dataset.')
# @click.option('--generate_visual',is_flag=True) 
# @click.option('--self_consistency',is_flag=True) 
def analyze_db_dataset(dataset_path: str, query: str, 
                    structure_type: str, 
                    generate_visual: bool, 
                    add_exploration_step: bool, 
                    self_consistency: bool,
                    generate_sc_answer: Optional[str] = True,
                    library_programs: Optional[Iterable[Program]] = None,
                    use_variable_filtered_exploration: bool = False,
                    oai_model: Optional[str] = "gpt-4",
                    **kwargs) -> Union[ExperimentResult, FailedExperimentResult, SelfConsistencyExperimentResult]:
    if not self_consistency:
        return complete_pass_db(dataset_path, query, structure_type, generate_visual, add_exploration_step, self_consistency, 
                                use_variable_filtered_exploration=use_variable_filtered_exploration,
                                library_programs=library_programs, oai_model=oai_model,
                                **kwargs)
    else:
        final_analyses = []
        sc_runs = []
        SELF_CONSISTENCY_ITERS = 5
        for _ in range(SELF_CONSISTENCY_ITERS):
            logger.info(f"Starting self-consistency iteration {_}.")
            experiment_result = complete_pass_db(dataset_path, query, structure_type, generate_visual, add_exploration_step, self_consistency, 
                                                 library_programs=library_programs, 
                                                 use_variable_filtered_exploration=use_variable_filtered_exploration,
                                                 oai_model=oai_model,
                                                 **kwargs)
            if hasattr(experiment_result, 'workflow'): # this is how you know it succeeded.
                # execution_results = process_workflow(retrieve_system_prompt(structure_type), workflow)
                # final_analysis = execution_results[-1]['response']
                analysis_representation = represent_analysis(experiment_result.workflow)
                # answer_dict = _extract_json(execution_results[-1]['response'])
                final_analyses.append(analysis_representation)
            sc_runs.append(experiment_result)
            logger.info(f"Completing self-consistency iteration {_}.")
        assert len(sc_runs) == SELF_CONSISTENCY_ITERS
        if not len(final_analyses) >= 1  and len(final_analyses) <= SELF_CONSISTENCY_ITERS:
            logger.error(f"No successful runs with self-consistency after {SELF_CONSISTENCY_ITERS} attempts.")
            return FailedSelfConsistencyExperimentResult(sc_runs)
        self_consistency_prompt = generate_self_consistency_prompt(final_analyses)
        if generate_sc_answer:
            try:
                sc_response = process_regular_prompt([], self_consistency_prompt)
            except openai.BadRequestError as e:
                # TODO: what should we do when the SC prompt is too long..?
                print(f"Error occurred: {e}.")
                ipdb.set_trace()
                # assert that the error is about exceeding context length.
                # assert "maximum context length" in e.message
                # sc_response = process_regular_prompt([], generate_self_consistency_prompt(final_analyses[:-1]))
        else:
            return SelfConsistencyExperimentResult(tuple(sc_runs), EMPTY_SC_ANSWER)
    # TODO: we don't account for {structure_type} in the function below.
    # if structure_type == 'dag':
    #     exploration_passback = lambda exploration_result: f"The output of the previous code is:\n{exploration_result}\n\n" +\
    #         f"Now, how would you go about performing an analysis of the query '{query}' using a flowmason graph?" +\
    #         " In the final answer, please write down a scientific hypothesis in natural language" +\
    #         " derived from the provided dataset, clearly stating the context of hypothesis (if any)" +\
    #         " , variables chosen (if any) and relationship between those variables (if any). Include implementations" +\
    #         " of any statistical significance tests (if applicable)." 
    # elif structure_type == 'loose':
    #     exploration_passback = lambda exploration_result: f"The output of the previous code is:\n{exploration_result}\n\n" +\
    #         f"Now, how would you go about performing an analysis of the query '{query}'?" +\
    #         " In the final answer, please write down a scientific hypothesis in natural language" +\
    #         " derived from the provided dataset, clearly stating the context of hypothesis (if any)" +\
    #         " , variables chosen (if any) and relationship between those variables (if any). Include implementations" +\
    #         " of any statistical significance tests (if applicable)."
    # else:
    #     raise ValueError("Unimplemented structure type.")

    # if not self_consistency:
    #     exploration_step = generate_exploration_prompt(dataset_path, query)
    #     rev_second_prompt_template = lambda exec_result: f"The output of the previous code is:" +\
    #         f"\n{exec_result}\n\nBased on this analysis output, what is your answer for the query '{query}'?" +\
    #         f"Ground your answer in the context of the analysis (including any metrics and statistical significance tests)." +\
    #         f"Keep your response short and to the point (1-3 sentences). Start your response with 'RESPONSE:'"
    #     workflow = [
    #                 CodeRequestPrompt(exploration_step), 
    #                 SynthResultPrompt(exploration_passback, True), 
    #                 SynthResultPrompt(rev_second_prompt_template, False)
    #                 ]
    #     return complete_pass_db(workflow, structure_type, "flowma_cache")
    # else:
    #     raise NotImplementedError("Self-consistency not implemented for database datasets.")
    #     first_message = construct_first_message(dataset_path, query, structure_type)
    #     rev_second_prompt_template = lambda exec_result: f"The output of the previous code is:\n{exec_result}\n\nBased on this output, what is your scientific hypothesis?"
    #     workflow = [CodeRequestPrompt(first_message), SynthResultPrompt(rev_second_prompt_template, False)] 

    if not self_consistency:
        execution_results = process_workflow(retrieve_system_prompt(structure_type), workflow)
        if add_exploration_step:
            display_results_explore_workflow(workflow, execution_results)
        elif not add_exploration_step:
            display_results_basic_workflow(workflow, execution_results)
        clear_dir("scratch/flowmason_cache")
    else: 
        SELF_CONSISTENCY_ITERS = 5
        final_analyses = []
        for _ in range(SELF_CONSISTENCY_ITERS):
            logger.info(f"Starting self-consistency iteration {_}.")
            try:
                execution_results = process_workflow(retrieve_system_prompt(structure_type), workflow)
                final_analysis = execution_results[-1]['response']
                final_analyses.append(final_analysis)
                logger.info(f"Completing self-consistency iteration {_}.")
            except ValueError as e:
                logger.error(f"Error occurred during self-consistency iteration {_}: {e}.")
            finally:
                clear_dir("scratch/flowmason_cache")
        print("=====Final analyses=====")
        for i in range(SELF_CONSISTENCY_ITERS):
            print(f"=====Iteration {i}=====")
            print(final_analyses)
        dataset_and_query_prompt = generate_dataset_description_and_query_only_prompt(dataset_path, query) 
        self_consistency_prompt = generate_self_consistency_prompt(dataset_and_query_prompt, final_analyses)
        ipdb.set_trace()
        sc_response = process_regular_prompt([], self_consistency_prompt)
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

def get_qr_data_split(use_test: bool, 
                      **kwargs):
    with open("data/qrdata/QRData.json", 'r') as file:
        qrdata = json.load(file)
    # split into train (10%) and test set (90%)
    # get the indices of the train and test set
    indices = list(range(len(qrdata)))
    random.Random(4).shuffle(indices)
    train_indices = indices[:len(indices) // 10]
    test_indices = indices[len(indices) // 10:]
    if use_test:
        return test_indices[:100]
    else:
        return train_indices

def _retrieve_individual_run_answer(workflow_results, index):
    return workflow_results[index].answer

def _retrieve_sc_run_answer(sc_results):
    agent_response = sc_results.final_answer['agent_response']
    try:
        # get the first integer in the response
        preferred_answer_i = int(re.search(r'\d+', agent_response).group())
    except ValueError:
        print(f"Error occurred: {agent_response}.")
        ipdb.set_trace()
    successful_runs = list(filter(lambda x: hasattr(x, 'answer'), sc_results.individual_runs))
    prediction = successful_runs[preferred_answer_i].answer
    return prediction

def compute_metrics_qrdata(indices: List[int], 
                           workflow_results: List[Dict[str, str]], 
                           self_consistency: bool,
                           **kwargs):
    def _filter_indices(bad_indices):
        # remove the indices at positions in bad_indices
        new_indices = []
        for i in range(len(indices)):
            if i in bad_indices:
                continue
            new_indices.append(indices[i])
        return new_indices
    predictions = []
    ground_truth_answers = []
    with open("data/qrdata/QRData.json", 'r') as file:
        qrdata = json.load(file)

    for i in range(len(workflow_results)):
        # check if 'answer' is in the attributes of workflow_results[i]
        index = indices[i]
        instance = qrdata[index]
        ground_truth_answers.append(instance['answer'])
        if ('answer' not in dir(workflow_results[i])) and ('final_answer' not in dir(workflow_results[i])):
            predictions.append("failed result")
        elif not self_consistency:
            predicted_answer = _retrieve_individual_run_answer(workflow_results, i)
            predictions.append(predicted_answer)
        else:
            predicted_answer = _retrieve_sc_run_answer(workflow_results, i)
            predictions.append(predicted_answer)
        print(f"=====Instance {i}=====")
        print(f"Predicted answer: {predicted_answer}")
        print(f"Ground truth answer: {instance['answer']}")
    # compute accuracy. For percentages, count +/- 3% as correct.
    num_correct = 0
    for i in range(len(predictions)):
        if isinstance(predictions[i], float) or isinstance(predictions[i], int): 
            try:
                ground_truth_answers[i] = float(ground_truth_answers[i])
                if abs(predictions[i] - ground_truth_answers[i]) <= 0.03:
                    num_correct += 1
            except ValueError:
                continue 
        elif isinstance(predictions[i], str) and isinstance(ground_truth_answers[i], str):
            if predictions[i].strip().lower() == ground_truth_answers[i].strip().lower():
                num_correct += 1
                continue
            elif len(ground_truth_answers[i]) == 1:
                if predictions[i][0].lower() == ground_truth_answers[i][0].lower():
                    num_correct += 1
                    continue
        elif predictions[i] == ground_truth_answers[i]:
            num_correct += 1
    logger.info(f"Accuracy: {num_correct/len(predictions)}")
    ipdb.set_trace()

    # compute the average number of errors in a workflow
    total_errors = sum([result.num_errors for result in workflow_results])
    avg_errors = total_errors / len(workflow_results)
    logger.info(f"Average number of errors: {avg_errors:.3f}")
    # what is the mode of the number of errors?
    error_counts = Counter([result.num_errors for result in workflow_results])
    mode_errors = max(error_counts, key=error_counts.get)
    # the most common number of errors
    logger.info(f"Mode of the number of errors: {mode_errors}")

def step_write_analysis_results_to_json(workflow_results: List[ExperimentResult], 
                                        structure_type: str,
                                        dataset: str,
                                        **kwargs):
    # need two more datapoints: exploration code
    # and the final analysis code
    def _extract_exploration_code(result: ExperimentResult):
        return extract_code(result.workflow[0]['response'])

    def _extract_exploration_output(result: ExperimentResult):
        return result.workflow[0]['execution_output']

    def _extract_analysis_code(result: ExperimentResult):
        return extract_code(result.workflow[1]['response'])

    def _extract_analysis_output(result: ExperimentResult):
        return result.workflow[1]['execution_output']

    def _construct_json_obj(result):
        analysis_code = _extract_analysis_code(result)
        exploration_code = _extract_exploration_code(result)
        exploration_output = _extract_exploration_output(result)
        analysis_output = _extract_analysis_output(result)
        return {
            # TODO: add question and ground-truth answer?
            "exploration_code": exploration_code,
            "exploration_output": exploration_output, 
            "analysis_code": analysis_code,
            "analysis_output": analysis_output,
            "answer": result.answer,
            "num_errors": result.num_errors,
            "num_prompt_tokens": result.num_prompt_tokens,
            "num_response_tokens": result.num_response_tokens,
            "complete_workflow": result.workflow
        }
    presentable_results = [_construct_json_obj(result) for result in workflow_results]
    fname = f"{dataset}_{structure_type}.json"
    with open(fname, 'w') as file:
        json.dump(presentable_results, file)
    return fname
    # with open("data/qrdata/analysis_results.json", 'w') as file:
    #     json.dump(workflow_results, file)

def sample_exploratory_analyses(results_fname: str, **kwargs):
    with open(results_fname, 'r') as file:
        results = json.load(file)
    # sample 5 exploratory analyses
    sample_indices = random.sample(range(len(results)), 5)
    for i in sample_indices:
        print(f"=====Exploratory analysis {i}=====")
        print(f"----Exploration code----\n {results[i]['exploration_code']}")
        print(f"----Exploration output----\n {results[i]['exploration_output']}")

def get_db_datasplit(use_test: bool) -> List[Tuple[str, str, str]]: # metadata paths and queries
    query_info = [] # will contain project name, metadata name, query, hypothesis
    if not use_test:  # train
        # loop over discoverybench/train
        for dirname in filter(lambda x: not x == 'nls_bmi_raw', os.listdir("discoverybench/real/train")):
            # iterate over all the metadata_{index}.json files in the directory
            for fname in filter(lambda x: x.endswith("json"), os.listdir(f"discoverybench/real/train/{dirname}")):
                with open(f"discoverybench/real/train/{dirname}/{fname}", 'r') as file:
                    metadata = json.load(file)
                    assert len(metadata['queries']) == 1 # accidentally double-nested?
                    for query_dict in metadata['queries'][0]:
                        qid = query_dict['qid']
                        query = query_dict['question']
                        hypothesis = query_dict['true_hypothesis']
                        query_info.append(
                            (
                                dirname, 
                                f"discoverybench/real/train/{dirname}/{fname}", 
                                query, 
                                hypothesis, 
                                qid
                            )
                        )
    else:
        raise ValueError("Unimplemented.")
    # filter out the "immigration_offshoring_effect_on_employment" project queries
    query_info = list(filter(lambda x: not x[0] == "immigration_offshoring_effect_on_employment", query_info))
    return query_info

def construct_db_dataset_frame(path: str, 
                               answer_key_path: str, 
                               use_answer_key: Optional[bool] = True) -> pl.DataFrame:
    # read all the metadata files 
    # each metadata file has 1 workflow and 1 or more queries
    # each query has a question, hypothesis, qid, and answer
    # return a dataframe with the columns: project_name, metadata_version, query, qid, answer/hypothesis, and workflow
    answer_key_frame = pl.read_csv(answer_key_path)
    rows = []
    project_name = os.path.basename(path)
    for fname in filter(lambda x: x.endswith('json'), os.listdir(path)):
        with open(f"{path}/{fname}", 'r') as file:
            try:
                metadata = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error occurred: {fname}.")
                raise e
            assert len(metadata['queries']) == 1
            for query_dict in metadata['queries'][0]:
                qid = query_dict['qid']
                query = query_dict['question']
                # hypothesis = query_dict['true_hypothesis']
                workflow = metadata['workflow']
                metadata_id = int(fname.split(".")[0].split("_")[1])
                metadata_path = f"{path}/{fname}"
                rows.append([project_name, metadata_id, query, qid, workflow, metadata_path])
                # TODO: randomly assign the `is_test` column.
    frame = pl.DataFrame(rows, schema=['dataset', 
                                       'metadataid', 
                                       'query', 
                                       'query_id', 
                                       'workflow', 
                                       'metadata_path' 
                                       ])
    is_test_vec = np.ones(len(frame))
    num_test_points = len(frame) // 2
    is_test_vec[:num_test_points] = 0
    rng = np.random.default_rng(seed=42) 
    # seed = np.random.RandomState()
    rng.shuffle(is_test_vec)
    # convert the is_test_vec to a boolean column
    is_test_vec = is_test_vec.astype(bool)
    frame = frame.with_columns([
        pl.Series(is_test_vec).alias('is_test')
    ])
    # create a column analysis id which is just dataset  metadataid and query_id joined by underscores
    frame = frame.with_columns([
        pl.concat_str(
            [
                frame['dataset'],
                frame['metadataid'],
                frame['query_id']
            ], 
            separator = "_"
        ).alias('analysis_id')
    ])
    frame = frame.join(answer_key_frame, on=['dataset', 'metadataid', 'query_id'])
    return frame

def step_provide_supervision_on_incorrect(
                                          dataset_path: str, query: str, 
                                          failure_memories: Iterable[str],
                                          structure_type: str, 
                                          generate_visual: bool, 
                                          add_exploration_step: bool, 
                                          self_consistency: bool,
                                          workflow: str,
                                          generate_sc_answer: bool,
                                          use_variable_filtered_exploration: bool = False,
                                          oai_model: str = "gpt-4",
                                          **kwargs):
    if len(failure_memories) == 0:
        return -1 
    else:
        workflow_hint_result = analyze_db_dataset(dataset_path, query,
                                                structure_type, generate_visual,
                                                add_exploration_step, self_consistency,
                                                generate_sc_answer=generate_sc_answer,
                                                nl_workflow=workflow, 
                                                failure_memories=failure_memories, 
                                                use_variable_filtered_exploration=use_variable_filtered_exploration,
                                                oai_model=oai_model)
        return workflow_hint_result

@click.command()
@click.option('--structure-type',
              type=click.Choice(['loose', 'functions', 'dag'], case_sensitive=False))
@click.option('--add_exploration_step', is_flag=True, help='Have the agent first explore the univariate distributions of the dataset.')
@click.option('--generate_visual',is_flag=True) 
@click.option('--self_consistency',is_flag=True) 
@click.option('--use_test', is_flag=True)
# @click.option('--index', type=int, help='The index of the QR data to analyze.', default=-1)
def execute_qrdata_map_dict(structure_type: bool, 
                            add_exploration_step: bool, 
                            generate_visual: bool, 
                            use_test: bool,
                            self_consistency: bool):
    qr_data_datapoint_dict = OrderedDict()
    qr_data_datapoint_dict['step_complete_datapoint'] = SingletonStep(analyze_qrdata, {
        "version": "001"
    })
    map_dict = OrderedDict()
    map_dict['map_step'] = MapReduceStep(qr_data_datapoint_dict, {
        # supply the parameters for the analyze_qrdata function here
        "index": get_qr_data_split(use_test)
    },
    {
        "structure_type": structure_type,
        "add_exploration_step": add_exploration_step,
        "generate_visual": generate_visual,
        "self_consistency": self_consistency,
        "generate_visual": False,
        "version": "001"
    }, list, 'index', [])
    map_dict['compute_metrics'] = SingletonStep(compute_metrics_qrdata, {
        "indices": tuple(get_qr_data_split(use_test)),
        "workflow_results": 'map_step',
        "version": "001", 
        "self_consistency": self_consistency
    })
    map_dict['write_analysis_results_to_json'] = SingletonStep(step_write_analysis_results_to_json, {
        "workflow_results": 'map_step',
        "structure_type": structure_type,
        "dataset": "qrdata",
        "version": "001"
    })
    map_dict['sample_exploratory_analyses'] = SingletonStep(sample_exploratory_analyses, {
        "results_fname": 'write_analysis_results_to_json', 
        "version": "001",
    })
    run_metadata = conduct(MY_CACHE_DIR, map_dict, MY_LOG_DIR)
    ipdb.set_trace()

def compute_metrics_db(experiment_result_claims, **kwargs):
    evaluation_labels = []
    for i in range(len(experiment_result_claims)):
        evaluation = experiment_result_claims[i][1]
        if isinstance(experiment_result_claims[i][1], dict):
            evaluation = repair_json(evaluation['agent_response'])
            evaluation_labels.append(eval(evaluation)['label'])
        elif isinstance(experiment_result_claims[i][1], str):
            assert experiment_result_claims[i][1] == "The experiment failed, so the claim cannot be evaluated."
            evaluation_labels.append("FAIL")
        else:
            raise ValueError("Unimplemented evaluation type.")
    print(pl.Series(evaluation_labels).value_counts())

    # print all of the analyses that were refuted
    # for i in range(len(experiment_result_claims)):
    #     if evaluation_labels[i] == "REFUTE":
    #         print(f"=====Refuted analysis {i}=====")
    #         print(f"=====Claim=====")
    #         print(experiment_result_claims[i][1])
    # # print all of the analyses that were supported
    # for i in range(len(experiment_result_claims)):
    #     if evaluation_labels[i] == "SUPP":
    #         print(f"=====Supported analysis {i}=====")
    #         print(f"=====Claim=====")
    #         print(experiment_result_claims[i][1])

    # # print all of the analyses that were partially supported
    # for i in range(len(experiment_result_claims)):
    #     if evaluation_labels[i] == "PRT_SUPP":
    #         print(f"=====Partially supported analysis {i}=====")
    #         print(f"=====Claim=====")
    #         print(experiment_result_claims[i][1])
    
    # # print all of the analyses that lacked information
    # for i in range(len(experiment_result_claims)):
    #     if evaluation_labels[i] == "LACK_INFO":
    #         print(f"=====Analysis that lacked information {i}=====")
    #         # print the query
    #         print(f"=====Query=====")
    #         print(experiment_result_claims[i][0])
    #         print(f"=====Claim=====")
    #         print(experiment_result_claims[i][1])

def step_combine_analysis_with_evaluation(experiment_result: ExperimentResult, 
                                         supervised_result: ExperimentResult,
                                         claim_evaluation: Union[str, Dict[str, str]],
                                         revised_evaluation: Union[str, Dict[str, str]],
                                         **kwargs) -> List[Tuple[ExperimentResult, str]]:
    if claim_evaluation == "The experiment failed, so the claim cannot be evaluated.":
        original_eval_label = "FAIL"
    else:
        original_eval_label = eval(repair_json(claim_evaluation['agent_response']))['label']

    if revised_evaluation == "The experiment failed, so the claim cannot be evaluated.":
        revised_eval_label = "FAIL"
    else:
        revised_eval_label = eval(repair_json(revised_evaluation['agent_response']))['label']
    
    if original_eval_label == "SUPP" or original_eval_label == "PRT_SUPP":
        return (experiment_result, claim_evaluation)
    elif revised_eval_label == "SUPP" or revised_eval_label == "PRT_SUPP":
        return (supervised_result, revised_evaluation)
    else:
        return (supervised_result, revised_evaluation)
    
def step_produce_contrastive_pairs(experiment_result: SelfConsistencyExperimentResult, 
                                   claims: Tuple,
                                   **kwargs) -> List[Tuple[ExperimentResult, ExperimentResult]]:

    successful_runs = []
    reason_fail_runs = []
    execution_fail_runs = []
    for i in range(len(experiment_result.individual_runs)):
        run_result = experiment_result.individual_runs[i]
        claim = claims[i]
        if not (isinstance(claim, dict)) and claim.startswith("The experiment failed, so the claim cannot be evaluated"):
            try:
                execution_fail_runs.append((run_result, DB_ANALYSIS_FAILED, "FAIL", run_result.tracebacks[-1]))
            except IndexError as e:
                logger.error("Error on retrieving the last traceback for a failed run.")
                ipdb.set_trace()
            continue
        label = eval(repair_json(claim['agent_response']))['label']
        rationale = eval(repair_json(claim['agent_response']))['rationale']
        analysis_output = run_result.workflow[-2]['execution_output']
        if label == "SUPP" or label == "PRT_SUPP":
            successful_runs.append((run_result, rationale, label, analysis_output))
        else:
            reason_fail_runs.append((run_result, rationale, label, analysis_output))
    if len(successful_runs) == 0: 
        return tuple([])
    if any([run[2] == "SUPP" for run in successful_runs]):
        successful_runs = list(filter(lambda x: x[2] == "SUPP", successful_runs)) 
    succ_fail_pairs = product(successful_runs, reason_fail_runs)
    contrast_examples = []
    for pair in succ_fail_pairs:
        succ_run, fail_run = pair
        if fail_run[2] == "FAIL":
            incorrect_program = fail_run[0].implementation_attempts[-1]
        else:
            incorrect_program = extract_code(fail_run[0].workflow[-2]['response'])
        correct_program = extract_code(succ_run[0].workflow[-2]['response'])
        correct_answer = succ_run[0].answer
        correct_rationale = succ_run[1]
        correct_output = succ_run[3]
        incorrect_rationale = fail_run[1] 
        incorrect_answer = fail_run[0].answer
        incorrect_output = fail_run[3]
        contrast_example = ContrastiveProgramExample(
            correct_program,
            correct_output,
            correct_answer, 
            correct_rationale,
            incorrect_program,
            incorrect_output,
            incorrect_answer,
            incorrect_rationale, 
            incorrect_rationale == DB_ANALYSIS_FAILED
        )
        contrast_examples.append(contrast_example)
    return tuple(contrast_examples)

def step_combine_analysis_with_evaluation_test(experiment_result: Union[ExperimentResult, FailedExperimentResult], 
                                               claim_evaluation: Union[str, Dict[str,str]], 
                                               **kwargs):
    return (experiment_result, claim_evaluation)

def step_compute_metrics_table(num_train_queries: int, num_test_queries: int, 
                               library_subroutines: Iterable[Tuple[str, str, str]],
                               lib_test_experiment_results: Iterable[Tuple[ExperimentResult, Dict]], 
                               no_lib_test_experiment_results: Iterable[Tuple[ExperimentResult, Dict]],
                               **kwargs): 
    num_correct_first_try = sum([s[-1] == 'first_try' for s in library_subroutines])
    num_correct_with_supervision = sum([s[-1] == 'resampled_with_supervision' for s in library_subroutines])
    ipdb.set_trace()
    logger.info(f"Number of training examples correct on first round of samples: {num_correct_first_try}/{num_train_queries}")
    logger.info(f"Number of training examples correct after resampling with supervision: {num_correct_with_supervision}/{num_train_queries}")

    num_correct_lib = sum([extract_db_label(summary[1]) in ['PRT_SUPP', 'SUPP'] for summary in lib_test_experiment_results]) 
    num_correct_no_lib = sum([extract_db_label(summary[1]) in [ 'PRT_SUPP', 'SUPP'] for summary in no_lib_test_experiment_results])
    logger.info(f"Number of test examples correct with library: {num_correct_lib}/{num_test_queries}")
    logger.info(f"Number of test examples correct without library: {num_correct_no_lib}/{num_test_queries}")

    number_of_lack_info_lib = sum([extract_db_label(summary[1]) == 'FAIL' for summary in lib_test_experiment_results])
    number_of_lack_info_no_lib = sum([extract_db_label(summary[1]) == 'FAIL' for summary in no_lib_test_experiment_results])
    logger.info(f"Number of examples that lacked information with library: {number_of_lack_info_lib}/{num_test_queries}")
    logger.info(f"Number of examples that lacked information without library: {number_of_lack_info_no_lib}/{num_test_queries}")

@click.command()
@click.option('--structure-type',
              type=click.Choice(['loose', 'functions', 'dag'], case_sensitive=False))
@click.option('--add_exploration_step', is_flag=True, help='Have the agent first explore the univariate distributions of the dataset.')
@click.option('--generate_visual',is_flag=True)
@click.option('--use_test', is_flag=True)
@click.option('--self_consistency',is_flag=True)
def execute_discovery_bench_map_dict(structure_type: bool, 
                                    add_exploration_step: bool, 
                                    generate_visual: bool, 
                                    use_test: bool,
                                    self_consistency: bool):
    discovery_bench_datapoint_dict = OrderedDict()
    query_infos = get_db_datasplit(use_test)
    
    def _get_db_identifier(query_info: Tuple[str, str, str, str, str]):
        proj_name = query_info[0]
        metadata_version = os.path.basename(query_info[1]).split(".")[0]
        qid = query_info[-1]
        return f"{proj_name}_{metadata_version}_qid_{qid}"

    def combine_analysis_with_evaluation(experiment_result, claim_evaluation, **kwargs) -> Tuple:
        return (experiment_result, claim_evaluation)

    queries = [query_info[2] for query_info in query_infos]
    metadata_paths = [query_info[1] for query_info in query_infos]
    hypotheses = [query_info[3] for query_info in query_infos]
    identifiers = [_get_db_identifier(query_info) for query_info in query_infos]

    mr_dict = OrderedDict()
    mr_dict['step_complete_datapoint'] = SingletonStep(analyze_db_dataset, {
        "version": "002"
    })
    # TODO: need to implement the evaluation prompt.
    mr_dict['step_evaluate_claim'] = SingletonStep(step_evaluate_claim, {
        "experiment_result": 'step_complete_datapoint',
        'version': '004'
    })
    mr_dict['step_combine_analysis_with_evaluation'] = SingletonStep(combine_analysis_with_evaluation, {
        "experiment_result": 'step_complete_datapoint',
        "claim_evaluation": 'step_evaluate_claim',
        "version": '001'
    })
    discovery_bench_datapoint_dict['map_step_train'] = MapReduceStep(mr_dict, {
        "dataset_path": metadata_paths,
        "query": queries,
        "gold_hypothesis": hypotheses,
        "db_identifier": identifiers
    },
    {
        "structure_type": structure_type,
        "add_exploration_step": add_exploration_step,
        "generate_visual": generate_visual,
        "self_consistency": self_consistency,
        "version": "001"
    }, list, 'db_identifier', []) # TODO: 'query_infos' needs to be changed.
    discovery_bench_datapoint_dict['compute_metrics'] = SingletonStep(compute_metrics_db, {
        "experiment_result_claims": 'map_step_train',
        "version": "001", 
    })
    output_dict = conduct(MY_CACHE_DIR, discovery_bench_datapoint_dict, MY_LOG_DIR)
    ipdb.set_trace()
    # load the library str

    # discovery_bench_datapoint_dict['step_complete'] = SingletonStep()

@click.command()
@click.argument('dataset_path')
@click.option('--structure-type',
                type=click.Choice(['loose', 'functions', 'dag'], case_sensitive=False))
@click.option('--add_exploration_step', is_flag=True, help='Have the agent first explore the univariate distributions of the dataset.')
@click.option('--use_variable_filtered_exploration', is_flag=True, help='Have the agent first explore the univariate distributions of the dataset.')
@click.option('--oai_model', type=click.Choice(['gpt-4', 'gpt-4o']))
@click.option('--answer_key_path', type=str, default="db_unsplit/answer_key_real.csv")
def execute_leap(dataset_path: str, structure_type: str, 
                use_variable_filtered_exploration: bool,
                 add_exploration_step: bool,
                 oai_model: str,
                 answer_key_path: str):
    frame = construct_db_dataset_frame(dataset_path, answer_key_path)
    # frame = construct_db_dataset_frame(dataset_path, "db_unsplit/answer_key_nls_bmi.csv")
    train_frame = frame.filter(pl.col('is_test')==False)
    test_frame = frame.filter(pl.col('is_test')==True)
    ll_steps_dict = OrderedDict()
    mapreduce_dict = OrderedDict()
    mapreduce_dict['step_sample_analyses'] = SingletonStep(analyze_db_dataset, {  # generate N samples.
        "version": "004",
        'self_consistency': True, 
        'generate_sc_answer': False,
        'use_variable_filtered_exploration': use_variable_filtered_exploration, 
        'oai_model': oai_model
    })
    mapreduce_dict['step_evaluate_samples'] = SingletonStep(step_evaluate_claim, { # evaluate each sample for whether it is correct or not.  N entailment labels 
        "experiment_result": 'step_sample_analyses',
        'eval_all_samples': True,
        'version': '001'
    })
    mapreduce_dict['step_generate_abstraction_responses'] = SingletonStep(step_write_standard_abstraction, { 
        'analysis_samples': 'step_sample_analyses',
        'sample_labels': 'step_evaluate_samples',
        'version': '003'
    })
    mapreduce_dict['step_generate_program_summaries_for_incorrect_samples'] = SingletonStep(step_describe_incorrect_programs, {
        'analysis_samples': 'step_sample_analyses',
        'sample_labels': 'step_evaluate_samples',
        'version': '006'
    })
    mapreduce_dict['step_resample_analyses_with_memory'] = SingletonStep(step_provide_supervision_on_incorrect, { 
        'failure_memories': 'step_generate_program_summaries_for_incorrect_samples',
        'self_consistency': True, 
        'oai_model': oai_model,
        'use_variable_filtered_exploration': use_variable_filtered_exploration,
        'generate_sc_answer': False, 
        'version': '001'
    })
    mapreduce_dict['step_evaluate_resampled_analyses'] = SingletonStep(step_evaluate_resampled_analysis, { # TODO: only for samples where all the questions were incorrect.
        'resampled_result': 'step_resample_analyses_with_memory',
        'eval_all_samples': True,
        'version': '002'
    })
    mapreduce_dict['step_generate_abstractions_for_resamples'] = SingletonStep(step_write_standard_abstraction, {
        'analysis_samples': 'step_resample_analyses_with_memory',
        'sample_labels': 'step_evaluate_resampled_analyses',
        'version': '003'
    })
    mapreduce_dict['step_collect_abstractions'] = SingletonStep(step_collect_abstractions, {
        'first_try_abstractions': 'step_generate_abstraction_responses',
        'resampled_abstractions': 'step_generate_abstractions_for_resamples',
        'version': '005'
    })
    ll_steps_dict['map_reduce_contrastive_program_generation'] = MapReduceStep(mapreduce_dict, {
        "dataset_path": train_frame['metadata_path'].to_list(),
        "query": train_frame['query'].to_list(),
        'gold_hypothesis': train_frame['gold_hypo'].to_list(),
        'analysis_id': train_frame['analysis_id'].to_list(),
        'workflow': train_frame['workflow'].to_list()
    },
    {
        'self_consistency': False, # NOTE: this is overriden for when we provide workflows. 
        'structure_type': structure_type,
        'add_exploration_step': add_exploration_step, 
        'generate_visual': False,
        'version': '001' 
    }, list, 'analysis_id', [])
    ll_steps_dict['collect_programs'] = SingletonStep(step_collect_programs, {
        'all_programs': 'map_reduce_contrastive_program_generation',
        'version': '011'
    })
    evaluation_mapreduce_dict = OrderedDict()
    evaluation_mapreduce_dict['step_complete_pass_w_library'] = SingletonStep(analyze_db_dataset, { 
        "version": "013",
        'library_programs': 'collect_programs',
        'use_variable_filtered_exploration': use_variable_filtered_exploration,
        'oai_model': oai_model
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
            'version': '002'
        },  list, 'analysis_id', []
    )
    evaluation_no_library_dict = OrderedDict()
    evaluation_no_library_dict['step_complete_pass_no_library'] = SingletonStep(analyze_db_dataset, {
        'version': '003',
        'use_variable_filtered_exploration': use_variable_filtered_exploration, 
        'oai_model': oai_model
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
    ll_steps_dict['compute_table_metrics'] = SingletonStep(step_compute_metrics_table, {
        'num_train_queries': len(train_frame),
        'num_test_queries': len(test_frame),
        'library_subroutines': 'map_reduce_contrastive_program_generation',
        'lib_test_experiment_results': 'map_reduce_test_w_library',
        'no_lib_test_experiment_results': 'map_reduce_test_no_library',
        'version': '001'
    })
    run_metadata = conduct(LEAP_CACHE_DIR, ll_steps_dict, MY_LOG_DIR)
    # program_summaries = load_artifact_with_step_name(run_metadata, 'map_reduce_contrastive_program_generation')
    # test_no_lib_examples = load_artifact_with_step_name(run_metadata, 'map_reduce_test_no_library')
    ipdb.set_trace()
    # 1. What is the main reason behind why one sample is incorrect but the other incorrect?
    # 2. Write a helper function that will help follow the reasoning of the correct program and thus avoid the incorrect program.
    # mapreduce_dict['']
    
@click.command()
@click.argument('dataset')
def create_train_real_answer_key(dataset):
    # dataset example: dataset=nls_bmi
    # dataset = dataset.split("=")[1]
    db_dataframe = load_db_dataframe(dataset)
    db_dataframe.write_csv(f'db_unsplit/answer_key_{dataset}.csv')
    # create a 
    # there are four columns we have to add for each datapoint: dataset,metadataid,query_id,gold_hypo

@click.command()
@click.argument('db_dataset')
@click.argument('query')
@click.argument('gold_hypothesis')
def run_individual_sc_query(db_dataset, query, gold_hypothesis):
    db_path = "db_unsplit"
    dataset_path = f"{db_path}/{db_dataset}"
    structure_type = 'loose'
    add_exploration_step = False
    step_dict = OrderedDict()
    step_dict['step_sample_analyses'] = SingletonStep(analyze_db_dataset, {  # generate N samples.
        'self_consistency': True, 
        'generate_sc_answer': False,
        'use_variable_filtered_exploration': True, 
        'oai_model': 'gpt-4o', 
        'query': query,
        'dataset_path': dataset_path,
        'structure_type': structure_type,
        'add_exploration_step': add_exploration_step, 
        'generate_visual': False,
        'version': '002'
    })
    step_dict['step_evaluate_samples'] = SingletonStep(step_evaluate_claim, { # evaluate each sample for whether it is correct or not.  N entailment labels 
        "experiment_result": 'step_sample_analyses',
        'eval_all_samples': True,
        'version': '001',
        'query': query,
        'gold_hypothesis': gold_hypothesis
    })
    step_dict['step_generate_abstraction_responses'] = SingletonStep(step_write_standard_abstraction, { 
        'analysis_samples': 'step_sample_analyses',
        'sample_labels': 'step_evaluate_samples',
        'query': query,
        'gold_hypothesis': gold_hypothesis,
        'version': '001'
    })
    run_metadata = conduct(INDIVIDUAL_QUERY_DIR, step_dict, MY_LOG_DIR)
    abstractions = load_artifact_with_step_name(run_metadata, 'step_generate_abstraction_responses')
    labels = load_artifact_with_step_name(run_metadata, 'step_evaluate_samples')
    ipdb.set_trace()
    

@click.group()
def main():
    pass



main.add_command(execute_qrdata_map_dict)
main.add_command(execute_discovery_bench_map_dict)
main.add_command(execute_leap)
main.add_command(create_train_real_answer_key)
main.add_command(run_individual_sc_query)
# main.add_command(execute_library_learning)

if __name__ == '__main__':
    main()