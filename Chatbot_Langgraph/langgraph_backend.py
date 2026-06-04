from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import BaseMessage, HumanMessage
from langsmith import tracing_context
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import requests
import random
import sqlite3
import os

os.environ["LANGSMITH_PROJECT"] = "Chatbot Project"  # Set the LangSmith project name

load_dotenv()  # Load environment variables from .env file

CONFIG = {'configurable': {'thread_id': 'thread_1'}}  # Example configuration, adjust as needed



# -------------------
# 1. LLM
# -------------------
llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), 
))

# --------------
# 2.Tools
# --------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=F4MDEAWICR7YTFFA"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState) -> ChatState:
    # take user query from state
    messages = state['messages']

    # send to llm
    response = llm_with_tools.invoke(messages)
    
    # response store state
    return {"messages": [response]}

tool_node = ToolNode(tools)


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
      with tracing_context(enabled=False):
        topic_response = llm.invoke([HumanMessage(content=topic_prompt)])
        topic = str(topic_response.content).strip().replace("\n", " ")
        return topic[:80] if topic else "New Chat"
      
    except Exception:
        return "New Chat"
    
# -------------------
# 5. SQLite Connection
# -------------------
conn= sqlite3.connect(database='chatbot.db', check_same_thread=False)

conn.execute(
    """
    CREATE TABLE IF NOT EXISTS thread_topics (
        thread_id TEXT PRIMARY KEY,
        topic TEXT NOT NULL
    )
    """
)
conn.commit()



# -------------------
# 6. Checkpointer
# -------------------
checkpointer = SqliteSaver(conn=conn)


# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)

# nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node,)

# edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')


# compile graph
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helper
# -------------------
def retrieve_all_threads():
    rows = conn.execute(
        """
        SELECT thread_id, MAX(rowid) AS last_seen
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY last_seen DESC
        """
    ).fetchall()

    return [thread_id for thread_id, _ in rows]


def save_thread_topic(thread_id: str, topic: str) -> None:
    conn.execute(
        """
        INSERT INTO thread_topics (thread_id, topic)
        VALUES (?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET topic = excluded.topic
        """,
        (thread_id, topic),
    )
    conn.commit()


def get_thread_topic(thread_id: str) -> str:
    row = conn.execute(
        "SELECT topic FROM thread_topics WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    return row[0] if row else ""


def retrieve_all_thread_topics() -> dict[str, str]:
    rows = conn.execute("SELECT thread_id, topic FROM thread_topics").fetchall()
    return {thread_id: topic for thread_id, topic in rows}
   








