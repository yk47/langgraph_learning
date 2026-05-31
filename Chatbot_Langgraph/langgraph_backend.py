from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import BaseMessage, HumanMessage
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


def generate_topic_from_messages(messages: list[BaseMessage]) -> str:
    """Generate a short conversation title from recent human/assistant messages."""
    if not messages:
        return "New Chat"

    recent_messages = messages[-8:]
    conversation_lines: list[str] = []

    for msg in recent_messages:
        msg_type = msg.__class__.__name__.replace("Message", "")
        conversation_lines.append(f"{msg_type}: {msg.content}")

    topic_prompt = (
        "Create a short topic title (max 6 words) for this conversation. "
        "Return only the title, no quotes, no punctuation at the end.\n\n"
        "Conversation:\n"
        + "\n".join(conversation_lines)
    )

    try:
        topic_response = llm.invoke([HumanMessage(content=topic_prompt)])
        topic = str(topic_response.content).strip().replace("\n", " ")
        return topic[:80] if topic else "New Chat"
    except Exception:
        return "New Chat"

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

# nodes
graph.add_node('chat_node', chat_node)

# edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile graph
chatbot = graph.compile(checkpointer=checkpointer)







