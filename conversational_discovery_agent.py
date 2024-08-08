import traceback
import polars as pl
import openai
import pandas as pd
from typing import Union, Tuple
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
from typing import List, Dict
import click
from io import StringIO
import json
import os
import dill

from packages.constants import SCRATCH_DIR, MY_CACHE_DIR, MY_LOG_DIR
from packages.json_repair import repair_json

@dataclass
class CodeRequestPrompt:
    prompt: str

@dataclass
class SynthResultPrompt:
    prompt_template: Callable[[str], str]
    is_code_req: bool

@dataclass
class ExperimentResult:
    status: str
    workflow: List[Dict]
    num_prompt_tokens: int
    num_response_tokens: int
    answer: Union[str, float]
    num_errors: int

@dataclass
class SelfConsistencyExperimentResult:
    individual_runs: Tuple[ExperimentResult]
    final_answer: str

class CodeExtractionException(Exception):
    def __init__(self, message, workflow_i, workflow):
        super().__init__(message)
        self.message = message
        self.workflow_so_far = workflow

# create an exception class CodeRetryException(Exception):
class CodeRetryException(Exception):
    def __init__(self, 
                 workflow_so_far,
                 workflow_i, message, prior_implementations, prior_tracebacks):
        self.workflow_so_far = workflow_so_far
        self.workflow_i = workflow_i
        self.message = message
        self.prior_implementations = prior_implementations
        self.prior_tracebacks = prior_tracebacks

class LongPromptException(Exception):
    def __init__(self, 
                 workflow_so_far,
                 workflow_i, message
                ):
        self.workflow_so_far = workflow_so_far
        self.workflow_i = workflow_i
        self.message = message




class FailedExperimentResult():
    def __init__(self, failed_phase_i, 
                 tracebacks, 
                 fail_type: str, 
                 implementation_attempts: List[str],
                 workflow_so_far):


        self.failed_phase_i = failed_phase_i
        self.tracebacks = tracebacks  # when teh fail type is a code retry exception
        self.implementation_attempts = implementation_attempts # when the fail type is a code retry exception
        self.workflow_so_far = workflow_so_far # when failed_phase_i is 0, this will be empty
        self.fail_type = fail_type
        

class ExtractJSONException(Exception):
    def __init__(self, response_object, message):
        self.response_object = response_object
        self.message = message


load_dotenv()
api_key = os.environ['THE_KEY']
client = OpenAI(api_key=api_key)  # TODO: put this in an env instead.
logger = loguru.logger

def dag_system_prompt(floma_cache):
    system_prompt = "You are a discovery agent who can execute a python code only once to answer a query based on one or more datasets. The datasets will be present in the current directory. Please write your code in the form of a `flowmason` directed acyclic graph: Here's an example of a flowmason graph:"
    floma_example = f"""
    ```
    from flowmason import SingletonStep, conduct, load_artifact_with_step_name  # make sure to import flowmason definitions
    from typing import Tuple
    from collections import OrderedDict
    CACHE_DIR="scratch/{floma_cache}"
    LOG_DIR="discoveryagent_logs"

    def _step_toy_fn(arg1: float, **kwargs) -> Tuple[float]: # all flowmason step functions must have **kwargs for supplying metadata
        return (3.1 + arg1, 3.1 + arg1)
    
    def _step_intransitive_fn(**kwargs) -> float:
        return 3.0

    def _step_toy_fn_two(arg1: Tuple[float, float], **kwargs) -> float:
        element1, element2 = arg1
        return element1 + element2
    
    def _step_toy_fn_three(arg1: float, arg2: float, arg3: float, **kwargs) -> float:
        return arg1 + arg2

    step_dict = OrderedDict()
    # NOTE: mutable objects (lists and dictionaries most commonly) cannot serve as arguments for flowmason steps (since they can't be pickled). Consider using tuples or json strings in those cases.
    # If you must use a mutable object (e.g. list or dict) for a python function, write a different function that can be called by the flowmason step function.
    step_dict['step_singleton'] = SingletonStep(_step_toy_fn, {{
        'version': "001", # don't forget to supply version, it is a mandatory argument
        'arg1': 2.9
    }})
    step_dict['step_intransitive'] = SingletonStep(_step_intransitive_fn, {{
        'version': "001" # required even when there are no arguments
    }})
    step_dict['step_singleton_two'] = SingletonStep(_step_toy_fn_two, {{
        'version': "001", 
        'arg1': 'step_singleton' # this will supply the return value of the previous step as an argument for 'arg1'
        # note that the return value of 'step_singleton' matches the type signature of arg1 for 'step_singleton_two'
    }})
    step_dict['step_singleton_three'] = SingletonStep(_step_toy_fn_three, {{
        'version': "001",
        'arg1': 'step_singleton',
        'arg2': 'step_intransitive', 
        'arg3': 3.0
    }})

    run_metadata = conduct(CACHE_DIR, step_dict, LOG_DIR) # this will run the flowmason graph, executing each step in order. It will substitute any step name parameters with the actual return values of the steps.
    # NOTE: if the step function has a print statement, it will be printed to the console. 
    # NOTE: if the step has no return statement, then loading an artifact will result in a FileNotFoundError. 
    # NOTE: when supplying a step name as an argument, make sure that the step has already been defined in the step_dict (otherwise you will just be passing a string).
    output_step_singleton_two = load_artifact_with_step_name(run_metadata, 'step_singleton_two')
    print(output_step_singleton_two) # 18.0. Make sure to print. Simply writing the variable name will not print the output.
    ``` 

    Answer the questions in the format:

    Observation: {{Full analysis plan in natural language}}.
    Code: 
    ```python
    {{All code to act on observation.}}
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

def retrieve_system_prompt(structure_type, floma_cache="flowmason_cache"):
    if structure_type == 'dag':
        return dag_system_prompt(floma_cache)
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
        raise CodeExtractionException("``` backticks not found", "n/a", "n/a")
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
            "\n\n Please provide a new and correct code implementation." + \
            "Remember to update the version number of the function if you update its implementation." +\
            "If there is a ModuleNotFoundError on an import statement, then it is mispelled or it is not installed (more likely). If it is not installed, you should not try to use it in the code." 
    return message


# TODO: implement retries for the code.
def process_code_step(history: List[Dict[str, str]], 
                            user_code_prompt, 
                            workflow_i: int,
                            current_code: str = "",
                            num_retries = 3) -> str:
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
                model="gpt-4",
                messages=messages,
                max_tokens=2000
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
            raise CodeExtractionException(f"Code could not be extracted from the response:\n\n{response} .", -1, "n/a")
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
        model="gpt-4",
        messages=messages,
        max_tokens=2000
    )
    num_prompt_tokens = response.usage.prompt_tokens
    num_response_tokens = response.usage.completion_tokens
    # return response.choices[0].message.content
    return {'agent_response': response.choices[0].message.content,
            'num_prompt_tokens': num_prompt_tokens,
            'num_response_tokens': num_response_tokens}

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
            try:
                result_dict = process_code_step(messages, workflow_step.prompt, i, "\n\n".join(all_code_snippets), num_retries=3)
            except openai.BadRequestError as e:
                raise LongPromptException(workflow_results, i, e.message, [], [])
            except openai.RateLimitError as e:
                raise LongPromptException(workflow_results, i, e.message, [], [])
            except CodeRetryException as e:
                raise CodeRetryException(workflow_results, i, e.message, e.prior_implementations, e.prior_tracebacks)
            except CodeExtractionException as e:
                raise CodeExtractionException(e.message, i, workflow_results)
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
            if workflow_step.is_code_req:
                try:
                    result_dict = process_code_step(messages, prompt, i, "\n\n".join(all_code_snippets), num_retries=3)
                except openai.BadRequestError as e:
                    raise LongPromptException(workflow_results, i, e.message)
                except openai.RateLimitError as e:
                    raise LongPromptException(workflow_results, i, e.message)
                except CodeRetryException as e:
                    raise CodeRetryException(workflow_results, i, e.message, e.prior_implementations, e.prior_tracebacks)
                except CodeExtractionException as e:
                    raise CodeExtractionException(e.message, i, workflow_results)
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

def generate_self_consistency_prompt(initial_q: str, final_answers: List[str]) -> str:
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

# def _extract_json(orig_response):
#     # the response might be of the form 
#     # ```json
#     # {
#     # ...
#     # }
#     # or just the json object itself.
#     response = orig_response.replace("\n", "")
#     final_response = response
#     try:
#         if response.startswith("`{"):
#             response = response[1:-1]
#             final_response = response
#         elif "```" not in response:
#             response = response.strip()
#             assert response.startswith("{") and response.endswith("}"), ipdb.set_trace()
#             final_response = response
#         else:
#             if response.startswith("```") and not response.endswith("```"): # see test case 2 for this function
#                 response = response + "```"
#             # find the second last occurrence of ``` in the response, since the last one is the closing backticks.
#             # s[:s.rfind("b")].rfind("b")
#             index = response.rfind("```", 0, response.rfind("```"))
#             response = response[index:]
#             ipdb.set_trace()
#             if "=" in response[index:]:
#                 index = response.find("{")
#                 end_index = response.rfind("}")
#                 final_response = response[index:end_index+1]
#             if response.startswith("```json"):
#                 start = response.find("```json") + len("```json")
#                 if response.endswith("```"):
#                     end = response.rfind("```")
#                     end = response.find("```", start)
#                 else:
#                     end = response.find("}") + 1
#                 final_response = response[start:end]
#             elif response.startswith("```python"):
#                 start = response.find("```python") + len("```python")
#                 end = response.find("```", start)
#                 final_response = response[start:end]
#             elif response.startswith("```{"):
#                 start = response.find("```") + len("```")
#                 end = response.find("```", start)
#                 final_response = response[start:end]
#             else:
#                 ipdb.set_trace()
#     except Exception as e:
#         logger.error(f"Error occurred: {e}.")
#         # TODO: try printing the stacktrace?
#         raise ExtractJSONException(response, "Error occurred during JSON extraction.")
#     try:
#         return eval(final_response)
#     except Exception as e:
#         logger.error(f"Error occurred: {e}.")
#         ipdb.set_trace()
#         raise ExtractJSONException(final_response, "Error occurred during eval call.")
def _extract_json(orig_response):
    try:
        start = orig_response.find("{")
        end = orig_response.rfind("}")
        
        if start == -1 or end == -1:
            raise ExtractJSONException(orig_response, "JSON object not found in the response.")
        
        json_str = orig_response[start:end+1]
        json_str = json_str.replace("\n", "").strip()
        return eval(json_str)
    
    except Exception as e:
        logger.error(f"Error occurred: {e}.")
        raise ExtractJSONException(orig_response, "Error occurred during JSON extraction.")


# def _extract_json(orig_response):
#     response = orig_response.replace("\n", "").strip()
    
#     try:
#         # Handle raw JSON string
#         if response.startswith("{") and response.endswith("}"):
#             final_response = response
        
#         # Handle JSON enclosed in triple backticks
#         elif "```" in response:
#             # Ensure closing backticks
#             if response.startswith("```") and not response.endswith("```"):
#                 response += "```"
            
#             # Extract content between the first and last triple backticks
#             start_index = response.find("```")
#             end_index = response.rfind("```")
#             final_response = response[start_index + 3:end_index].strip()
            
#             # Remove language indicators if present (e.g., ```json)
#             for lang in ["json", "python"]:
#                 if final_response.startswith(lang):
#                     final_response = final_response[len(lang):].strip()
        
#         # Handle JSON enclosed in single backticks
#         elif response.startswith("`{") and response.endswith("}`"):
#             final_response = response[1:-1]
        
#         else:
#             raise ExtractJSONException(response, "Unable to extract JSON")
        
#         return json.loads(final_response)
    
#     except Exception as e:
#         logger.error(f"Error occurred: {e}.")
#         raise ExtractJSONException(response, "Error occurred during JSON extraction.")


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

def complete_pass_db(workflow, structure_type, floma_cache):
    try:
        execution_results = process_workflow(retrieve_system_prompt(structure_type, floma_cache), workflow)
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
        # print(f"Error occurred: {e}.")
        # ipdb.set_trace()

    # TODO: implement build_experiment_result and cache the result. 
    # return build_experiment_result(execution_results, answer_dict)

def complete_pass_qrdata_db(workflow, structure_type):
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
        return complete_pass_qrdata_db(workflow, structure_type)
    else:
        SELF_CONSISTENCY_ITERS = 5
        final_analyses = []
        sc_runs = []
        for _ in range(SELF_CONSISTENCY_ITERS):
            logger.info(f"Starting self-consistency iteration {_}.")
            experiment_result = complete_pass_qrdata_db(workflow, structure_type) # either ExperimentResult or FailedExperimentResult
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

def step_evaluate_claim(experiment_result: Union[ExperimentResult, FailedExperimentResult], 
                        query: str,
                        gold_hypothesis, **kwargs) -> str: 
    print("At evaluation stage.") 
    if hasattr(experiment_result, 'answer'):
        predicted_answer = experiment_result.answer
        prompt = f"Consider the query: '{query}'. Then, Consider the following claim (C1):\n\n'{gold_hypothesis}'.\n\nThen, consider the analysis A1:\n\n'{predicted_answer}'\n\nDoes A1 support C1 (SUPP), partially support C1 (PRT_SUPP),  refute C1 (REFUTE), or not provide enough info (LACK_INFO)? Write your answer in the form of a json with two keys: label and rationale." 
        match_response = process_regular_prompt([], prompt)
        return match_response
    elif hasattr(experiment_result, 'final_answer'):
        predicted_answer = experiment_result.final_answer
        prompt = f"Consider the query: '{query}'. Then, Consider the following claim (C1):\n\n'{gold_hypothesis}'.\n\nThen, consider the analysis A1:\n\n'{predicted_answer}'\n\nDoes A1 support C1 (SUPP), partially support C1 (PRT_SUPP),  refute C1 (REFUTE), or not provide enough info (LACK_INFO)? Write your answer in the form of a json with two keys: label and rationale." 
        match_response = process_regular_prompt([], prompt)
        return match_response
    elif hasattr(experiment_result, 'fail_type'):
        return "The experiment failed, so the claim cannot be evaluated."
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
                    **kwargs) -> Union[ExperimentResult, FailedExperimentResult, SelfConsistencyExperimentResult]:
    # qr_dataset =  
    # the workflow is a list of the form [str, Function[str] -> str, Function[str] -> str, ...]
    # past_traces = retrieve_past_dags(extract_dataset_name(dataset_path))
    # identify the workflow to use:
    # if past_traces and not add_exploration_step:
    # if add_exploration_step:
    #     first_message = construct_first_message(dataset_path, query, structure_type, past_traces)
    if not self_consistency:
        exploration_step = generate_exploration_prompt(dataset_path, query)
        exploration_passback = lambda exploration_result: f"The output of the previous code is:\n{exploration_result}\n\n" +\
            f"Now, how would you go about performing an analysis of the query '{query}' using a flowmason graph?" +\
            " In the final answer, please write down a scientific hypothesis in natural language" +\
            " derived from the provided dataset, clearly stating the context of hypothesis (if any)" +\
            " , variables chosen (if any) and relationship between those variables (if any). Include implementations" +\
            " of any statistical significance tests (if applicable)." 
        rev_second_prompt_template = lambda exec_result: f"The output of the previous code is:" +\
            f"\n{exec_result}\n\nBased on this analysis output, what is your answer for the query '{query}'?" +\
            f"Ground your answer in the context of the analysis (including any metrics and statistical significance tests)." +\
            f"Keep your response short and to the point (1-3 sentences). Start your response with 'RESPONSE:'"
        workflow = [
                    CodeRequestPrompt(exploration_step), 
                    SynthResultPrompt(exploration_passback, True), 
                    SynthResultPrompt(rev_second_prompt_template, False)
                    ]
        return complete_pass_db(workflow, structure_type, "flowma_cache")
    else:
        first_message = construct_first_message(dataset_path, query, structure_type)
        rev_second_prompt_template = lambda exec_result: f"The output of the previous code is:\n{exec_result}\n\nBased on this output, what is your scientific hypothesis?"
        workflow = [CodeRequestPrompt(first_message), SynthResultPrompt(rev_second_prompt_template, False)] 

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

def _retrieve_sc_run_answer(sc_results, index):
    sc_result = sc_results[index]
    agent_response = sc_result.final_answer['agent_response']
    try:
        # get the first integer in the response
        preferred_answer_i = int(re.search(r'\d+', agent_response).group())
    except ValueError:
        print(f"Error occurred: {agent_response}.")
        ipdb.set_trace()
    if len(sc_result.individual_runs) == 10:
        # set individual runs to every even number
        individual_runs = sc_result.individual_runs[::2]
        assert len(individual_runs) == 5
    elif len(sc_result.individual_runs) == 5:
        individual_runs = sc_result.individual_runs
    else:
        raise ValueError("Unimplemented number of individual runs.")
    successful_runs = list(filter(lambda x: hasattr(x, 'answer'), sc_result.individual_runs))
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
    for i in range(len(experiment_result_claims)):
        if evaluation_labels[i] == "REFUTE":
            print(f"=====Refuted analysis {i}=====")
            print(f"=====Claim=====")
            print(experiment_result_claims[i][1])

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
        'version': '002'
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
    conduct(MY_CACHE_DIR, discovery_bench_datapoint_dict, MY_LOG_DIR)
    # discovery_bench_datapoint_dict['step_complete'] = SingletonStep()


@click.group()
def main():
    pass

main.add_command(execute_qrdata_map_dict)
main.add_command(execute_discovery_bench_map_dict)

if __name__ == '__main__':
    main()