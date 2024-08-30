from conversational_discovery_agent import parse_floma_dag, _extract_json
import ipdb
import pytest

@pytest.fixture
def simple_toy_block():
    code_block = """
    from flowmason import SingletonStep, conduct, load_artifact_with_step_name  # make sure to import flowmason definitions
    from collections import OrderedDict
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
    """
    yield code_block

@pytest.fixture
def experiment_block():
    code_block = """
    from flowmason import SingletonStep, conduct, load_artifact_with_step_name  # make sure to import flowmason definitions
    from collections import OrderedDict
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
    """
    yield code_block


def test_parse_floma_dag(simple_toy_block):
    result = parse_floma_dag(simple_toy_block)
    print(result)
    pass

def test_parse_floma_dag_2(experiment_block):
    result = parse_floma_dag(experiment_block)
    print(result)
    pass

def test_extract_json():
    response = "```python\nfinal_answer = {\n    'answer': 'categorical',\n    'rationale': 'The smoke variable contains two unique values - No and Yes. These are categorical values representing different categories of respondents smoking habits.'\n}\n\nprint(final_answer)\n```\nThis code will print the final answer and rationale in a JSON compatible dictionary format. The 'answer' key contains the final answer to the query 'Is the smoke variable numerical or categorical?', and the 'rationale' key provides an explanation for the answer."
    _extract_json(response)

def test_extract_json_two():
    response = '```json\n{\n  "answer": -27.35,\n  "rationale": "The Average Treatment Effect on the Treated (ATT) of Proposition 99 on cigarette sales is estimated to be -27.35 using a difference-in-differences (DiD) approach. This negative number indicates that Proposition 99 led to a decrease in cigarette sales."\n}\n'
    _extract_json(response)

def test_extract_json_three():
    response = '{"answer": "A", "rationale": "Based on the Granger Causality test that checks whether we can predict \'plcg\' using past values of \'praf\' better than just past values of \'plcg\' itself, the result suggests that \'praf\' \'Granger causes\' \'plcg\'."}'
    answer = _extract_json(response)
    print(answer)

def test_extract_json_four():
    response = '```\n{\n  "answer": "C",\n  "rationale": "The Pearson correlation coefficient between PIP2 and PIP3 is relatively low (approximately 0.195), suggesting that there is a weak linear relationship between the two variables. While correlation does not imply causation, a low correlation coefficient suggests that there is likely no causative relationship between PIP2 and PIP3."\n}\n```'
    return _extract_json(response)