import requests

# from langchain import OpenAI
from langchain.agents import initialize_agent, load_tools, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
import config

OPENAI_API_KEY = config.OPENAI_API_KEY

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0.8,
    model_name="gpt-3.5-turbo-16k-0613"
)

prompt = PromptTemplate(
  input_variables=["query"],
  template="You are New Native Internal Bot. Help users with their important tasks, like a professor in a particular field. Query: {query}"
)

search = DuckDuckGoSearchRun()

# Web Search Tool
search_tool = Tool(
    name = "Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions."
)

agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=[ search_tool],
    llm=llm,
    verbose=True, # I will use verbose=True to check process of choosing tool by Agent
    max_iterations=3
)

# llm_chain = LLMChain(llm=llm, prompt=prompt)
# print(llm_chain.run("What is lablab.ai"))
r_1 = agent("What is lablab.ai?")
print(f"Final answer: {r_1['output']}")