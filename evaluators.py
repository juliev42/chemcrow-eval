import os
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain

from dotenv import load_dotenv

load_dotenv()
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']




class BaseEvaluator():
    def __init__(self, llm = None):

        self.llm = llm if llm != None else OpenAI(temperature=0, model_name='gpt-3.5')
       
        self.prefix = """You are a chemistry professor evaluating your student's answer to a chemistry question. It is important to evaluate the factual correctness of their 
        You have access to the following tools:"""

        self.suffix = """Begin! Check if the answer addresses the question accurately and completely. 

        Question: {question}
        Student's Answer: {answer}
        {agent_scratchpad}"""

        self.tools = self.get_tools()
        self.prompt = ZeroShotAgent.create_prompt(
            self.tools, prefix=self.prefix, suffix=self.suffix, input_variables=["question", "answer" "agent_scratchpad"]
        )
        self.llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=self.prompt)
        self.agent = ZeroShotAgent(llm_chain=self.llm_chain, tools=self.tools)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
        agent=self.agent, tools=self.tools, verbose=True
        )
    
    def get_tools(self):
        return []
    
    def run(self, question, answer, agent_scratchpad=None):
        return self.agent_executor.run(question=question, answer=answer, agent_scratchpad=agent_scratchpad)
    


class SearchEvaluator(BaseEvaluator):
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






SearchEvaluator().run(question="What is the capital of California?", answer="Sacramento")


    



