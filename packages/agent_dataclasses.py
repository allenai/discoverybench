from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple, Union, Optional

@dataclass
class CodeRequestPrompt:
    prompt: str

@dataclass
class SynthResultPrompt:
    prompt_template: Callable[[str], str]
    is_code_req: bool
    has_code_provision: Optional[bool] = False

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
    def __init__(self, message, workflow_i, workflow, agent_response):
        super().__init__(message)
        self.workflow_i = workflow_i
        self.message = message
        self.workflow_so_far = workflow
        self.agent_response = agent_response

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
                 workflow_so_far, 
                 last_agent_response: Optional[str] = None):


        self.failed_phase_i = failed_phase_i
        self.tracebacks = tracebacks  # when teh fail type is a code retry exception
        self.implementation_attempts = implementation_attempts # when the fail type is a code retry exception
        self.workflow_so_far = workflow_so_far # when failed_phase_i is 0, this will be empty
        self.fail_type = fail_type
        self.last_agent_response = "" if last_agent_response is None else last_agent_response

class FailedSelfConsistencyExperimentResult():
    def __init__(self, failed_runs: List[FailedExperimentResult]):
        self.failed_runs = failed_runs

        

class ExtractJSONException(Exception):
    def __init__(self, response_object, message):
        self.response_object = response_object
        self.message = message

@dataclass
class ContrastiveProgramExample():
    correct_program: str
    correct_output: str
    correct_answer: str
    correct_eval_rationale: str
    incorrect_program: str
    incorrect_output: str
    incorrect_answer: str
    incorrect_eval_rationale: str
    is_execution_failure: bool

@dataclass
class Program():
    signature: str
    summary: str
    program: str
    query: str
    imports: List[str]