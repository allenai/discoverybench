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