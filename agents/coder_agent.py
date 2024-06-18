import pdb
from dotenv import load_dotenv
from agents.coder_utils import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_together import Together
from langchain_openai import ChatOpenAI
import os
import json
from utils.dv_log import DVLogger
import uuid
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler

# uncomment the following line to enable debug mode
# langchain.debug = True

load_dotenv()


def get_prompt_data(
        prompt_config: str = None
):
    if prompt_config is None and os.environ.get("PROMPT_CONFIG") is None:
        raise ValueError("PROMPT_CONFIG not set and prompt_config not provided")
    else:
        prompt_config = prompt_config or os.environ.get("PROMPT_CONFIG")

    with open(prompt_config, "r") as file:
        return json.load(file)


class BaseAgent():
    def __init__(
        self,
        model_config: str = None,
        api_config: str = None,
        model_name: str = "gpt-4-1106-preview",
        log_file: str = "output.log",
        max_iterations: int = 5
    ):
        self.logfile = log_file
        # logger.add(log_file, format="{time} {level} {message}", level="INFO")
        self.logger = DVLogger(f"{model_name}_{uuid.uuid4()}", log_file)
        self.file_handler = FileCallbackHandler(self.logfile)
        self.stdout_handler = StdOutCallbackHandler()
        self.max_iterations = max_iterations

        # check if model config is provided
        if model_config is None and os.environ.get("MODEL_CONFIG") is None:
            raise ValueError("MODEL_CONFIG not set and model_config not provided")
        else:
            # override environment variable config path if model_config is provided
            model_config = model_config or os.environ.get("MODEL_CONFIG")

        # do a similar check for api_config
        if api_config is None and os.environ.get("API_CONFIG") is None:
            raise ValueError("API_CONFIG not set and api_config not provided")
        else:
            api_config = api_config or os.environ.get("API_CONFIG")

        # load model config
        self.model_name = model_name
        with open(model_config, "r") as file:
            self.model_config = json.load(file)

        # load api config
        with open(api_config, "r") as file:
            self.api_config = json.load(file)

        # get full model name and type
        try:
            self.full_model_name = self.model_config['models'][self.model_name]['model_name']
            self.model_type = self.model_config['models'][self.model_name]['model_type']
        except KeyError:
            raise ValueError(f"Model {model_name} not found in model config")

        try:
            # get api key using model type
            self.api_key = self.api_config[self.model_type]
            if self.api_key == "":
                raise ValueError(f"API key for {self.model_type} in api config is empty")
        except KeyError:
            raise ValueError(f"API key for {self.model_type} not found in api config")

        # get the model
        self.llm = self.get_model(
            api=self.model_type,
            model=self.full_model_name,
            api_key=self.api_key
        )

        # create agent

        self.agent = create_agent(
            llm=self.llm,
            handlers=[self.file_handler, self.stdout_handler],
            max_iterations=self.max_iterations
        )

    def get_model(
            self,
            api,
            api_key,
            model,
            **kwargs
    ):
        llm = None
        if (api == "together"):
            llm = Together(
                model=model,
                together_api_key=api_key,
                **kwargs
            )
        elif (api == "anthropic"):
            llm = ChatAnthropic(
                model=model,
                api_key=api_key,
                **kwargs
            )
        elif (api == "openai"):
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                **kwargs
            )
        # elif (api == "google"):
        #     llm = ChatGoogleGenerativeAI(
        #         model=model,
        #         google_api_key=api_key,
        #         **kwargs
        #     )
        else:
            raise ValueError(f"Invalid API: {api}")
        return llm

    def generate(self, dataset_paths, query):
        system_prompt = """You are a discovery agent who can execute a python code only once to answer a query based on one or more datasets. The datasets will be present in the current directory. Please write your code in the form of a `flowmason` directed acyclic graph: Here's an example of a flowmason graph:

        from flowmason import Singletonstep, conduct 
        CACHE_DIR="scratch/flowmason_cache"
        LOG_DIR="discoveryagent_logs"

        def _step_toy_fn(arg1: float, **kwargs):
            print(arg1)
            return 3.1 + arg1

        step_dict = OrderedDict()
        step_dict['step_singleton'] = SingletonStep(_step_toy_fn, {
            'version': "001", 
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
        try:
            output = self.agent.invoke(input={
                "system_prompt": system_prompt,
                "input": f"Load all datasets using python using provided paths. Paths: {dataset_paths}. {query}"
            })
            self.logger.log_json(output)
        except Exception as e:
            print("Execution Stopped due to : ", e)
            self.logger.logger.error(f"Execution Stopped due to : {e}")
        self.logger.close()
