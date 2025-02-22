from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from agent import create_react_agent

load_dotenv()

db = SQLDatabase.from_uri("postgresql://al:12345@localhost:5432/postgres")
# llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatOllama(model="llama3.2", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

system_message = prompt_template.format(dialect="postgresql", top_k=5)
# print(prompt_template.messages[0].pretty_print())

agent_executer = create_react_agent(llm, tools, prompt=system_message)

question = "Which customer has bought most products?"

for step in agent_executer.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
