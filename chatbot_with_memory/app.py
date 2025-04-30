import os
import streamlit as st
import time
from datetime import datetime

import google.generativeai as genai

from src.utils import ChatbotConfig
from src.chatbot import EnhancedChatbot


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

config = ChatbotConfig(google_api_key=GOOGLE_API_KEY)
chatbot = EnhancedChatbot(config)

# Set page config
st.set_page_config(
    page_title="Chat Application",
    page_icon="💬",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .user-message {
        background-color: #e6f7ff;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        text-align: right;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    .assistant-message {
        background-color: #f0f0f0;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        display: flex;
        flex-direction: column;
    }
    .timestamp {
        font-size: 0.8em;
        color: #888;
        margin-top: 5px;
    }
    .chat-container {
        margin-bottom: 20px;
        max-height: 70vh;
        overflow-y: auto;
        padding: 10px;
    }
    .message-content {
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("💬 Chat Application")

# Initialize chat history in session state if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to add a message to the chat history
def add_message(role, content):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({"role": role, "content": content, "timestamp": timestamp})

# Function to display chat messages
def display_chat():
    with st.container():
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-content">{message["content"]}</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-content">{message["content"]}</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)

# User input section
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    display_chat()
    st.markdown('</div>', unsafe_allow_html=True)

    # Input elements arranged horizontally with send button
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("Type a message...", key="user_input", placeholder="Type your message here...")
    
    with col2:
        send_button = st.button("Send")

# Process user input when the send button is clicked
if send_button and user_input:

    add_message(role='user', content=user_input.lower())  

    # Simulate processing (you can replace this with actual AI response logic)
    with st.spinner("Thinking..."):

        response = chatbot.generate_response(user_input)
        add_message(role="assistant", content=response['response'])
        
    st.experimental_rerun()

# Add a clear chat button
if st.button("Clear Chat"):
    chatbot.clear_memory()
    st.session_state.chat_history = []
    st.experimental_rerun()

