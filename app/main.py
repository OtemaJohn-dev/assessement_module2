"""
This module handles user authentication and session management.
"""

import streamlit as st
from dotenv import dotenv_values
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import chromadb
from datetime import datetime
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
config = dotenv_values(".env")

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=config["AZURE_ENDPOINT_EMBEDDINGS"],
    api_key=config["AZURE_API_KEY_EMBEDDINGS"],
)
ef = embeddings.embed_documents


groq_api_key = config["GROQ_API_KEY"]

# server connection

client = chromadb.HttpClient(host="localhost", port=8000)

if not (ret := client.heartbeat()):
    st.error("Failed to connect to ChromaDB. Please check your configuration.")
    st.stop()
else:
    st.write(
        f"Connected to ChromaDB successfully.{
            datetime.fromtimestamp(int(ret/1e9))
        }"
             )
# Initialize ChromaDB collection
collection = client.get_or_create_collection("war_and_peace")


# Define initial system message
system_message = SystemMessage(
    content="""
    You are a helpful assistant that answers questions based on the context
    provided in the chromadb collection named 'war_and_peace'.
    If the context does not contain enough information to answer the question,
    you should say "I don't have enough information to answer that question."
    You should not make up answers or provide information that is not in the
    context.
    """
)

# Sidebar content
with st.sidebar:
    st.title("RAG Assistant")
    st.write(
        "Use this assistant to query your database content intelligently."
             )
    st.write("------------")
    if st.button("Clear Chat"):
        st.session_state.messages = [system_message]
        st.rerun()

# Initialize LLM
if "llm" not in st.session_state:
    llm = ChatGroq(
        model=config["GROQ_API_MODEL"],
        api_key=groq_api_key,
        temperature=0.1,
        max_tokens=131072,
    )
    st.session_state.llm = llm
else:
    llm = st.session_state.llm

# Page title
st.title("RAG system with Groq LLM")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [system_message]

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)


# Generate a response from the model
def give_response(msg: str):
    collection.query()
    messages = st.session_state.messages
    response = llm.invoke(messages)
    st.session_state.messages.append(response)
    return response

# Chat input


if msg := st.chat_input("Enter your question here:"):
    st.session_state.messages.append(HumanMessage(content=msg))
    with st.spinner("Thinking..."):
        response = give_response(msg)
    st.rerun()
