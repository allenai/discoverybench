import loguru
import ipdb

from .constants import DB_ANALYSIS_FAILED
from typing import Dict, Union
from .json_repair import repair_json

from .agent_dataclasses import CodeExtractionException

logger = loguru.logger
def extract_db_label(claim_evaluation: Union[str, Dict[str, str]]):
    if isinstance(claim_evaluation, str) and claim_evaluation in DB_ANALYSIS_FAILED:
        return "FAIL"
    else:
        try:
            response = repair_json(claim_evaluation['agent_response'])
        except TypeError:
            ipdb.set_trace()
        label = eval(response)['label']
        return label

def extract_db_rationale(claim_evaluation: Union[str, Dict[str, str]]):
    if claim_evaluation == DB_ANALYSIS_FAILED:
        return DB_ANALYSIS_FAILED
    else:
        response = repair_json(claim_evaluation['agent_response'])
        rationale = eval(response)['rationale']
        return rationale 


def extract_code(response):
    # the code should be encapsulated with ```python and ```.
    # extract the code from the response and return it as a multiline string.
    # find the first occurence of ```python
    try:
        response = response[response.index("```"): ]
    except ValueError:
        ipdb.set_trace()
        logger.error(f"``` backticks not found; response: {response}")
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
