import streamlit as st
from langgraph_backend import (
    chatbot,
    generate_topic_from_messages,
    retrieve_all_threads,
    retrieve_all_thread_topics,
    save_thread_topic,
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ******************************************* utility functions *****************************************
def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

def reset_chat():
    previous_thread_id = st.session_state.get('thread_id')
    thread_id = generate_thread_id()
    add_thread(thread_id) # Add the new thread ID to the list of chat threads
    if previous_thread_id:
        promote_thread_to_top(previous_thread_id)
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    st.session_state['thread_topics'][thread_id] = ""

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def promote_thread_to_top(thread_id):
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
        st.session_state['chat_threads'].insert(0, thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # This function can be implemented to load conversation history based on thread_id
    # For demonstration, we are just returning an empty list
    return state.values.get('messages', [])


def fallback_topic_from_messages(messages):
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = str(msg.content).strip().replace("\n", " ")
            if text:
                return text[:40] + ("..." if len(text) > 40 else "")
    return ""

# ***************************************** Session Setup *****************************************
# st.session_state is a dictionary-like object that allows you to store information across different runs of the app.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []



if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads() 

if 'thread_topics' not in st.session_state:
    st.session_state['thread_topics'] = retrieve_all_thread_topics()

add_thread(st.session_state['thread_id'])  # Add the current thread ID to the list of chat threads
st.session_state['thread_topics'].setdefault(st.session_state['thread_id'], "")



# ***************************************** Sidebar UI *****************************************
st.sidebar.title("LangGraph Chatbot")



if st.sidebar.button("New Chat", key="new_chat_btn", type="primary"):
    reset_chat()

# Custom CSS: only style the primary sidebar button (New Chat).
st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #16a34a !important;
        color: #ffffff !important;
        border: 1px solid #15803d !important;
    }

    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #15803d !important;
        border: 1px solid #166534 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("My Conversations")


for thread_id in st.session_state['chat_threads']:  
    if thread_id == st.session_state['thread_id']:
        continue

    topic_label = st.session_state['thread_topics'].get(thread_id, "").strip()
    if topic_label in ("", "New Chat"):
        messages = load_conversation(thread_id)
        topic_label = fallback_topic_from_messages(messages)

        # Hide threads that have no messages yet.
        if not topic_label:
            continue

        # Keep a temporary fallback label in memory, but do not overwrite the saved LLM topic.
        st.session_state['thread_topics'][thread_id] = topic_label
    if st.sidebar.button(topic_label, key=f"thread_btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:                
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages

        

# ***************************************** Main UI *****************************************
# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input("Type your message here...")

if user_input:
    # first add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)
    

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}  # Example configuration, adjust as needed

    # then add assistant response to history (for demonstration, we are just echoing the user input)
   
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Generate/update a human-readable topic for this thread.
    thread_messages = load_conversation(st.session_state['thread_id'])
    topic = generate_topic_from_messages(thread_messages)
    st.session_state['thread_topics'][st.session_state['thread_id']] = topic
    save_thread_topic(st.session_state['thread_id'], topic)
    promote_thread_to_top(st.session_state['thread_id'])