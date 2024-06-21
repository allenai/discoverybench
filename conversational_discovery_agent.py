from dotenv import load_dotenv
import sys
import pdb
from openai import OpenAI
from flowmason import conduct, load_artifact_with_step_name
import click
from io import StringIO
import json
import os

load_dotenv()
api_key = os.environ['THE_KEY']
client = OpenAI(api_key=api_key)  # TODO: put this in an env instead.

system_prompt = "You are a discovery agent who can execute a python code only once to answer a query based on one or more datasets. The datasets will be present in the current directory. Please write your code in the form of a `flowmason` directed acyclic graph: Here's an example of a flowmason graph:"
floma_example = """
```
from flowmason import SingletonStep, conduct, load_artifact_with_step_name  # make sure to import flowmason definitions
CACHE_DIR="scratch/flowmason_cache"
LOG_DIR="discoveryagent_logs"

def _step_toy_fn(arg1: float, **kwargs): # all flowmason step functions must have **kwargs for supplying metadata
    print(arg1)
    return 3.1 + arg1

step_dict = OrderedDict()
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

Observation: {idea in natural language}.
Code: 
```python
{code to act on observation.}
```
"""
full_system_prompt = system_prompt + floma_example



def construct_first_message(dataset_path, query):
    # TODO load the columns dataset_description = f"Consider the dataset at {dataset_path}. It contains the following columns: {columns}."
    # get the path containing the dataset
    path = os.path.dirname(dataset_path)
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
        columns = eval(f"{dataset['datasets'][0]['columns']['raw']}") # List[Dict] contains 'name' and 'description' keys.
        datapath = f"{path}/{dataset['datasets'][0]['name']}" # path to the csv"
    column_explanations = "\n".join([f"{column['name']}: {column['description']}" for column in columns])
    query_expanded = f"'{query}'"
    query_expanded = query_expanded + f" Here are the explanations of the columns in the dataset:\n\n {column_explanations}\n"
    return f"Suppose you had a dataset saved at {datapath} and you wanted to answer the following query: {query_expanded}\nHow would you go about performing this analysis using a flowmason graph? In the final answer, please write down a scientific hypothesis in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any) including any statistical significance."

def extract_code(response):
    # the code should be encapsulated with ```python and ```.
    # extract the code from the response and return it as a multiline string.
    # find the first occurence of ```python
    start = response.find("```python") + len("```python")
    # find the first occurence of ```
    end = response.find("```", start)
    # return the code
    return response[start:end]


@click.command()
@click.argument('dataset_path')
@click.argument('query')
def analyze_dataset(dataset_path: str, query: str):
    first_message = construct_first_message(dataset_path, query)
    first_response = client.chat.completions.create(
        model="gpt-4", 
        messages = [
            {"role": "system", "content": full_system_prompt},
            {'role': 'user', "content": first_message}
        ],
        max_tokens = 2000
    )
    response = first_response.choices[0].message.content
    extracted_code = extract_code(response)
    stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        # NOTE: by default, loguru, which is the primary logger for flowmason, logs to stderr.
        exec(extracted_code, locals(), locals()) # I'm not sure why locals() is needed here, but it doesn't work without it.
        printed_output = sys.stdout.getvalue()
        # provide the printed output to the model, so it can construct the final message.
        second_response = client.chat.completions.create(
            model="gpt-4", 
            messages = [
                {"role": "system", "content": full_system_prompt},
                {'role': 'user', "content": printed_output},
                {'role': 'assistant', "content": response},
                {'role': 'user', "content": f"The output of the previous code is:\n{printed_output}\n\nBased on this output, what is your scientific hypothesis?"}
            ],
            max_tokens = 2000
        )
        hypothesis = second_response.choices[0].message.content
        sys.stdout = stdout
        print("=====Agent observation + code=====")
        print(response)
        print("=====Agent execution output=====")
        print(printed_output)
        print("=====Agent hypothesis=====")
        print(hypothesis)
    except Exception as e:
        sys.stdout = stdout
        print(f"An error occurred: {e}")
        pdb.set_trace()
    finally: 
        if sys.stdout != stdout:
            sys.stdout = stdout
        # if printed_output:
        #     print("Captured output:")
        #     print(printed_output)
        # else:
        #     print("No output was captured.")

if __name__ == '__main__':
    analyze_dataset()