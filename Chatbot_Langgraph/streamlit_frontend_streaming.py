import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
# st.session_state is a dictionary-like object that allows you to store information across different runs of the app.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

message_history = st.session_state['message_history']

CONFIG = {'configurable': {'thread_id': 'thread_1'}}  # Example configuration, adjust as needed

# loading the conversation history
for message in message_history:
    with st.chat_message(message['role']):
        st.text(message['content'])

#message_history.append({'role': 'user', 'content': 'Hi'})
#message_history.append({'role': 'assistant', 'content': 'Hello! How can I assist you today?'})

user_input = st.chat_input("Type your message here...")

if user_input:
    # first add user message to history
    message_history.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)
    

    
    # then add assistant response to history (for demonstration, we are just echoing the user input)
   
    with st.chat_message("assistant"):
       ai_message = st.write_stream(
           message_chunk for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
           )
       )
    message_history.append({'role': 'assistant', 'content': ai_message})