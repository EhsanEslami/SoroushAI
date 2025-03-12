import streamlit as st
import os
from dotenv import load_dotenv
from graph import query_pipeline  # Import the new pipeline function

load_dotenv()

st.title("Welcome to Souroush AI")
chat_placeholder = st.empty()

def init_chat_history():
    """Initialize chat history with a system message."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

def start_chat():
    """Start the chatbot conversation."""
    # Display chat messages from history on app rerun
    with chat_placeholder.container():
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("چه شعری دز ذهن داری؟"):
        # Append user query as a human message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response using the new pipeline
        response = query_pipeline(prompt)
        
        # Display assistant's response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Append assistant's response to the chat history
        st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    init_chat_history()
    start_chat()
