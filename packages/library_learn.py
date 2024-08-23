import re
import ipdb
import os
import loguru
import ipdb
from dotenv import load_dotenv
from typing import List
from openai import OpenAI
from .agent_dataclasses import ExperimentResult, FailedExperimentResult, ContrastiveProgramExample, SelfConsistencyExperimentResult, Program
from .answer_extract_utils import extract_db_label, extract_code, extract_db_rationale
from .json_repair import repair_json

from typing import Union, Tuple, Dict, Optional, Iterable

load_dotenv()
api_key = os.environ['THE_KEY']
client = OpenAI(api_key=api_key)  # TODO: put this in an env instead.
logger = loguru.logger

def extract_all_code(experiment_result):
    if hasattr(experiment_result, 'workflow'):
        code_str = ""
        explore_code = extract_code(experiment_result.workflow[0]['response'])
        analysis_code = extract_code(experiment_result.workflow[1]['response'])
    elif hasattr(experiment_result, 'final_answer'):
        best_workflow_index = int(experiment_result.final_answer['agent_response'][1])
        assert isinstance(best_workflow_index, int) 
        best_experiment_result = experiment_result.individual_runs[best_workflow_index]
        explore_code = extract_code(best_experiment_result.workflow[0]['response'])
        analysis_code = extract_code(best_experiment_result.workflow[1]['response'])
    return f"{explore_code}\n\n{analysis_code}"


def step_write_leap_helper(contrastive_pairs: Iterable[ContrastiveProgramExample], gold_hypothesis: str, 
                           query: str, **kwargs):
    responses = []
    for pair in contrastive_pairs:
        # 1. What is the main reason behind why one sample is incorrect but the other incorrect?
        # 2. Write a helper function that will help follow the reasoning of the correct program and thus avoid the incorrect program.
        correct_code = pair.correct_program
        incorrect_code = pair.incorrect_program
        correct_rationale = pair.correct_eval_rationale
        correct_output = pair.correct_output

        incorrect_rationale = pair.incorrect_eval_rationale
        incorrect_output = pair.incorrect_output
        prompt = f"Consider the following query about a dataset: {query}\n\n" +\
            f"The correct answer is {gold_hypothesis}.\n\n" +\
            f"Here are two programs that were written to answer the query:\n\n" +\
            f"Correct program:\n{correct_code}\n\n" +\
            f"Correct output: {correct_output}\n\n" +\
            f"Incorrect program:\n{incorrect_code}\n\n" +\
            f"Incorrect output: {incorrect_output}\n\n" +\
            f"The first program is right -- '{correct_rationale}'.\n\n" +\
            f"The second program is wrong -- '{incorrect_rationale}'.\n\n" +\
            f"Where do you think the second program went wrong and the first program went right?" +\
            f"Write a helper function that will help follow the reasoning of the correct program and thus avoid the incorrect program, so that it can be used in future analyses about this dataset." +\
            f"Put the function in a code block delineated by ```python and ```. You can add an example usage at the bottom of the block, but make sure it is commented out."
            # TODO: consider asking for a return statement.
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        response = response.choices[0].message.content
        # helper = extract_code(response) 
        responses.append(response)
    return tuple(responses) # TODO: delete the helpers that have the same function name

# TODO: this has to be modified. 
def step_write_standard_abstraction(analysis_samples: SelfConsistencyExperimentResult, 
                                    sample_labels: Iterable[str],
                                    query: str, 
                                    gold_hypothesis: str,
                                    **kwargs):
    if analysis_samples == -1:
        return tuple([])
    assert len(analysis_samples.individual_runs) == len(sample_labels)
    # get the successful programs
    responses = []
    successful_indices = []
    for i in range(len(sample_labels)):
        if extract_db_label(sample_labels[i]) == 'SUPP':
            successful_indices.append(i)
    if len(successful_indices) == 0:
        for i in range(len(sample_labels)):
            if extract_db_label(sample_labels[i]) == 'PRT_SUPP':
                successful_indices.append(i) 
    for i in range(len(successful_indices)):
        index = successful_indices[i]
        analysis = analysis_samples.individual_runs[index].workflow[1]
        correct_program = extract_code(analysis['response'])
        prompt = f"Consider the following query about a dataset: {query}\n\n" +\
            f"The correct answer is {gold_hypothesis}.\n\n" +\
            f"Here is a program that was written to answer the query:\n\n" +\
            f"{correct_program}\n\n" +\
            f"Generate one or more helper function from this program that can be re-used for future queries and analyses about this dataset." +\
            f"Put the function in a code block delineated by ```python and ```. You can add an example usage at the bottom of the block, but make sure it is commented out."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        response = response.choices[0].message.content
        responses.append(response)
    return tuple(responses)

def step_extract_helper(leap_helpers: Iterable[str], **kwargs):
    return tuple([extract_code(response) for response in leap_helpers])

# TODO: we should sample multiple of these and pick the one with the longest fn.
def create_ll_prompt(queries: Tuple[str], experiment_result_claims: Tuple[Union[ExperimentResult, FailedExperimentResult], Dict], 
                     meta_reason_samples: Optional[bool] = False, **kwargs):
    succ_queries = []
    successful_experiment_results = []
    for i in range(len(experiment_result_claims)):
        if extract_db_label(experiment_result_claims[i][1]) in ['SUPP', 'PRT_SUPP']:
            succ_queries.append(queries[i])
            successful_experiment_results.append(experiment_result_claims[i])
    
    successful_code = [extract_all_code(x[0]) for x in successful_experiment_results]
    # print each succesful code block
    max_analyses = 10
    # succ_queries = succ_queries[max_analyses:]
    # successful_code = successful_code[max_analyses:]
    # succ_queries = succ_queries[:max_analyses]
    # successful_code = successful_code[:max_analyses]
    prompt_query_str = ""
    for i in range(len(succ_queries)):
        prompt_query_str += f"Query {i+1}: {succ_queries[i]}\n\nCode {i+1}:\n{successful_code[i]}\n\n"
    # NOTE: if we ask for more rewrites, then less useful helpers might be generated.
    prompt = f"Here are queries and associated code for {len(succ_queries)} data analyses." +\
        f"\n\n{prompt_query_str}" + \
        f"Please rewrite any 2 of the programs to be more efficient. The resulting programs must execute to the same result as the original program." +\
        f"Start by writing helper functions that can reduce the size of the code." +\
        f"Try to write helpers that can be used for multiple analyses." +\
        "Please format your answer as:\n" +\
        "// once at beginning \n" + \
        "NEW HELPERS:\n" +\
        "// repeated for i\n"+\
        "NEW CODE (i):\n" + \
        "You can assume that the helpers in NEW HELPERS will be defined for the new programs, so the implementations of the helpers should not be repeated in the new programs."
    # write prompt to a file, prompt.txt
    with open('prompt.txt', 'w') as f:
        f.write(prompt)
    print(prompt) 
    if not meta_reason_samples:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        response = response.choices[0].message.content
        # write response to a file, response.txt
        with open('response.txt', 'w') as f:
            f.write(response)
        helper_fn_lines = _extract_helpers(response)
        print(helper_fn_lines)
        return helper_fn_lines
    else:
        library_strs = []
        for _ in range(5):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            response = response.choices[0].message.content
            # write response to a file, response.txt
            helper_fn_lines = _extract_helpers(response)
            library_strs.append(helper_fn_lines)
        # now, prompt the model to select the best library string
        all_library_str = ""
        for i in range(len(library_strs)):
            all_library_str += f"Library {i+1}:\n{library_strs[i]}\n\n"
        print(all_library_str)
        prompt = f"Here are {len(library_strs)} helper function libraries:\n\n" +\
            f"{all_library_str}" +\
            f"These libraries were made for answering the following queries :\n\n" +\
            f"{"\n".join([f'{i}. {queries[i]}' for i in range(len(queries))])}\n\n" +\
            "Please the one library that you think is most useful for answering future queries of this type on the same dataset." +\
            f"Please return your answer in the form of a json object with the key 'selected_library_index' (0-based) and 'rationale' (str)"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        response = response.choices[0].message.content
        response_index = eval(repair_json(response))['selected_library_index']
        ipdb.set_trace()
        return library_strs[response_index]

def step_prettyprint_helper(contrastive_pairs: Iterable[ContrastiveProgramExample], 
                            leap_response: str, 
                            leap_helper: str,
                            standard_helper: str,
                            dataset_path: str,
                            analysis_id: str,
                            query: str,
                            gold_hypothesis: str,
                            **kwargs):
    """
    """
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    if len(contrastive_pairs) == 0:
        # remove the helpers file, if it exists
        if os.path.exists(f"{dataset_name}_{analysis_id}_helpers.txt"):
            os.remove(f"{dataset_name}_{analysis_id}_helpers.txt")
    else:
        f = open(f"{dataset_name}_{analysis_id}_helpers.txt", "w")
        # write the contrastive pair to the file
        correct_program_first = contrastive_pairs[0].correct_program
        correct_output = contrastive_pairs[0].correct_output
        correct_answer = contrastive_pairs[0].correct_answer

        incorrect_program_first = contrastive_pairs[0].incorrect_program
        incorrect_output = contrastive_pairs[0].incorrect_output
        incorrect_answer = contrastive_pairs[0].incorrect_answer
        # write the contrastive pair to the file
        f.write(f"Contrastive pair 1\n\n")
        f.write(f"Query:\n{query}\n\n")
        f.write(f"Gold hypothesis:\n{gold_hypothesis}\n\n")
        f.write(f"Correct program:\n{correct_program_first}\n\n")
        f.write(f"Correct output:\n{correct_output}\n\n")
        f.write(f"Correct answer:\n{correct_answer}\n\n")

        f.write(f"Incorrect program:\n{incorrect_program_first}\n\n")
        f.write(f"Incorrect output:\n{incorrect_output}\n\n")
        f.write(f"Incorrect answer:\n{incorrect_answer}\n\n")

        f.write(f"Leap response:\n\n")
        # write the leap response to the file
        f.write(f"{leap_response[0]}\n\n")


        # write the helper function to the file
        f.write(f"LEAP HELPER FUNCTION:\n\n")
        f.write(f"{leap_helper[0]}\n")

        # write the standard helper function to the file
        f.write(f"STANDARD HELPER FUNCTION:\n\n")
        f.write(standard_helper[0])

        # for i in range(len(contrastive_pairs)):
        #     for pair in contrastive_pairs:
        #         f.write(f"Contrastive pair {i+1}\n\n")
        f.close()

def _extract_helpers(response):
    # extract the helpers from the response
    # the helpers are the code blocks that start with NEW HELPERS:
    # and end with NEW CODE (i):
    helpers = []
    response = response.split("\n")
    in_helper = False
    for line in response:
        if  "NEW HELPERS:" in line:
            in_helper = True
            continue
        if "NEW CODE" in line:
            in_helper = False
            break
        if in_helper:
            helpers.append(line)
    return '\n'.join(helpers)

def step_describe_incorrect_programs(analysis_samples: SelfConsistencyExperimentResult,
                                    sample_labels: Iterable[str],
                                    query: str,
                                     **kwargs) -> Iterable[str]:
    # iterate through the labeled experiments, only returning a non-empty iterable if all of the labels are incorrect.
    # otherwise, we should return an empty iterable.
    incorrect_indices = []
    for i in range(len(sample_labels)):
        if extract_db_label(sample_labels[i]) in ['FAIL', 'LACK_INFO', 'REFUTE']:
            incorrect_indices.append(i)
    if len(incorrect_indices) < len(sample_labels):
        return tuple([])
    # TODO: what to do if there is a failure?
    memory_traces = []
    for i in range(len(incorrect_indices)):
        if extract_db_label(sample_labels[incorrect_indices[i]]) == 'FAIL':
            # TODO: figure this out
            analysis = analysis_samples.individual_runs[incorrect_indices[i]]
            program = analysis.implementation_attempts[-1]
            traceback = analysis.tracebacks[-1]
            prompt = f"Consider the following query: {query}\n\n" +\
                f"Here is a program that was written to answer the query:\n\n" +\
                f"{program}\n\n" +\
                f"However, the program failed to execute. Here is the traceback:\n\n" +\
                f"{traceback}\n\n" +\
                f"Please provide a 1-sentence summary of what the program does."
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            response = response.choices[0].message.content
            memory_trace = f"{response}\n" + \
                f"However, the program failed to execute, with the following traceback:\n\n" +\
                f"{traceback}"
            memory_traces.append(memory_trace)
        elif extract_db_label(sample_labels[incorrect_indices[i]]) in ['LACK_INFO', 'REFUTE']:
            analysis = analysis_samples.individual_runs[incorrect_indices[i]].workflow[1]
            program = extract_code(analysis['response'])
            prompt =f"Consider the following query: {query}\n\n" +\
                f"Here is a program that was written to answer the query:\n\n" +\
                f"{program}\n\n" +\
                f"Please provide a 1-sentence summary of what the program does."
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            response = response.choices[0].message.content
            rationale = extract_db_rationale(sample_labels[incorrect_indices[i]])
            memory_trace = f"{response}\n" +\
                f"However, the hypothesis deduced from the execution of " + \
                f"the program (A1) did not match the expected hypothesis (C1): {rationale}"
            memory_traces.append(memory_trace)
        else:
            ipdb.set_trace()
            raise ValueError("The label is not FAIL or LACK_INFO")
    return tuple(memory_traces)

def step_collect_abstractions(first_try_abstractions: Iterable[str],
                              resampled_abstractions: Iterable[str],
                              **kwargs):
    # collect all the abstractions
    if len(first_try_abstractions) > 0:
        return (first_try_abstractions[0])
    elif len(resampled_abstractions) > 0:
        return (resampled_abstractions[0])
    else:
        return ''
# def step_generate_abstracted_programs(: str, **kwargs):
#     ipdb.set_trace()

def _extract_function_signature(program):

    # function_signature_regex = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\):"
    # function_signature_regex = r'def\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)'
    function_signature_regex = r'(def\s+[a-zA-Z_]\w*\s*\([^)]*\))'
    first_sentence_regex = r'\"\"\"\n\s*(.+?)(?:\.|\n|$)'
    function_signature = re.search(function_signature_regex, program).groups()
    first_sentence = re.search(first_sentence_regex, program).groups()
    function_signatures = []
    for i in range(len(function_signature)):
        function_signatures.append((function_signature[i], first_sentence[i]))
    return function_signatures
    # write a regex for extracting the first sentence of the docstring, for example
    #
    # def gender_disparity_in_wealth(dataframe, year, metric='median'):
    #   """
    #   Computes gender disparity in wealth for individuals ever incarcerated in the specified year.
    #   
    #   Args:
    #     dataframe: pd.DataFrame
    #     year: int
    #     metric: str
    #   """
    # make sure to watch out for the indentation of the docstring.


def format_program_descriptions(library_programs: Iterable[Program]):
    return "\n".join([f"({i}) {program.signature}: {program.summary}" for i, program in enumerate(library_programs)])

def parse_library_selection_response(query: str, library_programs: Iterable[Program], agent_selection_response: str):
    json_response = eval(repair_json(agent_selection_response))
    indices = json_response['indices']
    program_str = "\n\n".join([library_programs[i].program for i in indices])
    program_backticked = f"```\n{program_str}\n```"

    elicit_prompt_str = "Here are the full definitions of the selected programs:\n\n" +\
        f"{program_backticked}\n" +\
        f"Now perform a statistically-principled analysis to answer the original query '{query}'" + \
        f"optionally using the selected programs."
    print(elicit_prompt_str)
    return elicit_prompt_str 

def step_collect_programs(all_programs: Iterable[str], **kwargs):
    programs = []
    for program in all_programs:
        if program == '':
            logger.warning("Empty program")
            continue
        program_code = extract_code(program)
        signature_tups = _extract_function_signature(program_code)
        for signature, first_sentence in signature_tups:
            programs.append(
                Program(signature=signature, summary=first_sentence, program=program_code)
            )
    return programs