from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

CONFIG = {'configurable': {'thread_id': 'thread_1'}}  # Example configuration, adjust as needed

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), 
))

def chat_node(state: ChatState) -> ChatState:
    # take user query from state
    messages = state['messages']

    # send to llm
    response = llm.invoke(messages)
    
    # response store state
    return {"messages": [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

# nodes
graph.add_node('chat_node', chat_node)

# edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile graph
chatbot = graph.compile(checkpointer=checkpointer)





