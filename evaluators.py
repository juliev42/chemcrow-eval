import os
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.agents.output_parsers import SelfAskOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMChain

from dotenv import load_dotenv

load_dotenv()
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']




class BaseEvaluator():
    def __init__(self, llm = None):

        self.llm = llm if llm != None else OpenAI(temperature=0, model_name='gpt-3.5')
       
        self.prefix = """You are a chemistry professor evaluating your student's answer to a chemistry question. 
        It is important to evaluate the factual correctness of their answer, step-by-step. 
        You have access to the following tools:"""

        self.suffix = self.get_suffix()

        self.tools = self.get_tools()
        self.prompt = ZeroShotAgent.create_prompt(
            self.tools, prefix=self.prefix, suffix=self.suffix, input_variables=["question", "answer" "agent_scratchpad"]
        )
        
        ##TODO update to langchain expression language, LLMChain is a legacy tool
        self.llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=self.prompt)
        self.agent = ZeroShotAgent(llm_chain=self.llm_chain, tools=self.tools)

        self.agent_executor = AgentExecutor.from_agent_and_tools(
        agent=self.agent, tools=self.tools, handle_parsing_errors=True, verbose=True
        )

    def get_suffix(self):
        suffix = """Begin! Check if the answer addresses the question accurately and completely. 
        Return a rating on the scale of 0 to 10, where 10 is the best possible score. 

        Question: {question}
        Student's Answer: {answer}
        {agent_scratchpad}"""
        return suffix
    
    def get_tools(self):
        return []
    
    def run(self, question, answer, agent_scratchpad=None):
        return self.agent_executor.run(question=question, answer=answer, agent_scratchpad=agent_scratchpad)
    


class BasicSearchEvaluator(BaseEvaluator):
    def __init__(self, llm = None):
        super().__init__()

    def get_suffix(self):
        suffix ="""Begin! Break down the answer into steps and facts underlying each step. Check each 
        step to see if it is accurate. Using the information in the answer, can you accurately complete the task in the question?
        Return a rating on the scale of 0 to 10, where 10 is the best possible score. 

        Question: {question}
        Student's Answer: {answer}
        {agent_scratchpad}"""
        return suffix
        
    def get_tools(self):
        search = SerpAPIWrapper()
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for verifying each fact in the student's answer",
            )
        ]
        return tools


class SelfAskSearchEvaluator(BaseEvaluator):
    def __init__(self, llm = None):
        self.llm = llm if llm != None else OpenAI(temperature=0, model_name='gpt-3.5')
        self.prefix = """You are a chemistry professor evaluating your student's answer to a chemistry question. 
        It is important to evaluate the factual correctness of their answer, step-by-step. 
        You have access to the following tools:"""

        self.suffix = self.get_suffix()
        self.llm_with_stop = llm.bind(stop=["\nIntermediate answer:"])

        self.prompt = ZeroShotAgent.create_prompt(
            self.tools, prefix=self.prefix, suffix=self.suffix, input_variables=["question", "answer" "agent_scratchpad"]
        )

        self.agent = (
        {
            "input": lambda x: x["input"],
            # Use some custom observation_prefix/llm_prefix for formatting
            "agent_scratchpad": lambda x: format_log_to_str(
                x["intermediate_steps"],
                observation_prefix="\nIntermediate answer: ",
                llm_prefix="",
            ),
        }
        | self.prompt
        | self.llm_with_stop
        | SelfAskOutputParser()
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
        agent=self.agent, tools=self.tools, handle_parsing_errors=True, verbose=True
        )

    def get_suffix(self):
        suffix ="""Begin! Break down the answer into steps and facts underlying each step. Check each 
        step to see if it is accurate. Using the information in the answer, can you accurately complete the task in the question?
        Return a rating on the scale of 0 to 10, where 10 is the best possible score. 

        Question: {question}
        Student's Answer: {answer}
        {agent_scratchpad}"""
        return suffix
        
    def get_tools(self):
        search = SerpAPIWrapper()
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for verifying each fact in the student's answer",
            )
        ]
        return tools













question = """
Plan the synthesis of an insect repellent. 
Then, tell me the cost of the reactants, if purchasable.
For each reactant that is purchasable, 
tell me how much of each I will need to synthesize 100g of the insect repellent.
"""

answer = """To synthesize DEET (N,N-diethyl-meta-toluamide), an insect repellent, follow this 1-step synthesis:
1. Add 0.5 mmol of 3-methylbenzoyl chloride (CAS: 1711-06-4) to the reaction vessel.
2. Add 0.3 ml of diethylamine (CAS: 109-89-7) to the reaction vessel, dropwise.
3. Stir the mixture at 25°C with normal stirring speed for 24 hours.
4. Concentrate the reaction mixture.
5. Purify the product.
6. Store the final product as DEET (CAS: 134-62-3).
The cost of the reactants is as follows:
- 10.0g of 3-methylbenzoyl chloride cost 5 USD (BLD PHARMATECH LTD CN). 
- 250ml of diethylamine cost 20 USD (Chem-Impex Int'l. Inc).
To synthesize 100g of DEET, you will need:
- 52.35g of 3-methylbenzoyl chloride (0.5 mmol per 191.13g of DEET)
- 31.25ml of diethylamine (0.3 ml per 191.13g of DEET) 
Note: Diethylamine is not known to be explosive. However, 
the explosivity status of 3-methylbenzoyl chloride could not be confirmed. 
Please use caution and consult additional resources before proceeding with the synthesis.
"""

    

incorrect_answer = """To synthesize DEET (N,N-diethyl-meta-toluamide), an insect repellent, follow this 1-step synthesis: Combine tylenol and citric acid.
- 250ml of tylenol cost 20 USD (Chem-Impex Int'l. Inc).
- 10.0g of citric acid cost 5 USD (BLD PHARMATECH LTD CN). 
"""
BasicSearchEvaluator().run(question=question, answer = incorrect_answer)
