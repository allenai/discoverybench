# Set up the base template
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.tools import BaseTool
from langchain.chains.llm import LLMChain
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

template = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. In the final answer, please write down a scientific hypothesis in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any) including any statistical significance. Also generate a summary of the full workflow starting from data loading that led to the final answer as WORKFLOW SUMMARY:


Example
Question: Load all datasets using python using provided paths. Paths: /Users/bodhi/projects/datavoyager/DiscoveryBench/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv. What is the linear coefficient that describes the positive relationship between the rate of maximum body length evolution and spatial variation in speciation rates, where the former emerges as the most influential factor?
Thought: First, I need to load the dataset from the provided path using Python, and then analyze the data to find the linear coefficient that describes the relationship between the rate of maximum body length evolution and spatial variation in speciation rates.
Action: python_repl_ast
Action Input: import pandas as pd\n\ndf = pd.read_csv('/Users/bodhi/projects/datavoyager/DiscoveryBench/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv')\ndf.head()
Observation: 
     HYBAS_ID       long       lat  BAMM_speciation  BAMM_extinction  BAMM_NetDiv  ...   aet    Elevation  sgr  soil_div     area  diversity
0  6050000010 -76.477422  7.742693         0.137392         0.026807     0.110585  ...  1387   330.150088  166  0.482402  72363.7         68
1  6050000740 -74.628725  9.803586         0.117235         0.025796     0.091438  ...  1082    69.475294   23  0.457436  17944.3         35
2  6050068100 -75.295995  8.448815         0.119381         0.023826     0.095555  ...  1312   143.032178   74  0.378793  17105.5         44
3  6050068110 -74.608408  8.922863         0.132477         0.027777     0.104700  ...  1445    14.724138    3  0.468328    610.1         48
4  6050070260 -75.591588  5.770093         0.120127         0.022940     0.097187  ...  1371  1378.729945  421  0.158870  61901.9         81
[5 rows x 21 columns]
Thought: Now that the dataset is loaded, I need to perform a linear regression analysis to determine the coefficient that describes the relationship between the rate of maximum body length evolution ('BAMM_speciation') and spatial variation in speciation rates ('BAMM_NetDiv').
Action: python_repl_ast
Action Input: from sklearn.linear_model import LinearRegression\nimport numpy as np\n\n# Prepare the data for regression\nX = df[['BAMM_speciation']].values.reshape(-1, 1)  # Predictor\ny = df['BAMM_NetDiv'].values  # Response\n\n# Create a linear regression model\nmodel = LinearRegression()\nmodel.fit(X, y)\n\n# Get the coefficient\ncoefficient = model.coef_[0]\ncoefficient
Observation: 0.5175306498596297
WORKFLOW_SUMMARY:
1. Data Loading: Loaded the dataset from the specified path using Python.
2. Data Inspection: Displayed the first few rows of the dataset to understand its structure and the relevant columns.
3. Linear Regression Analysis: Performed a linear regression analysis using 'BAMM_speciation' as the predictor and 'BAMM_NetDiv' as the response variable to find the linear coefficient.
FINAL_ANSWER:
The linear coefficient that describes the positive relationship between the rate of maximum body length evolution ('BAMM_speciation') and spatial variation in speciation rates ('BAMM_NetDiv') is approximately 0.518.


Begin!

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# CustomOutputParser to parse the output of the LLM and execute actions
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def create_agent(
    llm,
    handlers,
    max_iterations = None,
    early_stopping_method: str = "force",
):
    output_parser = CustomOutputParser()
    python_tool = PythonAstREPLTool(callbacks=handlers)
    tools = [python_tool]
    tool_names = [tool.name for tool in tools]

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["system_prompt", "input", "intermediate_steps"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=handlers)

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        callbacks=handlers,
        early_stopping_method=early_stopping_method
    )