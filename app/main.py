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

# Load Azure embeddings
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=config["AZURE_ENDPOINT_EMBEDDINGS"],
    api_key=config["AZURE_API_KEY_EMBEDDINGS"],
)


# Initialize ChromaDB client
client = chromadb.HttpClient(host="localhost", port=8000)

# Check ChromaDB connection
if not (ret := client.heartbeat()):
    st.error(" Failed to connect to ChromaDB.")
    st.stop()
else:
    st.success(
        f" Connected to ChromaDB at {datetime.fromtimestamp(int(ret / 1e9))}"
    )

# Get or create collection
collection = client.get_or_create_collection("romeo_and_juliet")

# Initial system instruction
system_message = SystemMessage(
    content="""
You are a helpful assistant that answers questions based on the context
provided in the ChromaDB collection named 'war_and_peace'.
If the context does not contain enough information to answer the question,
you should say "I don't have enough information to answer that question."
Never make up answers or include information not found in the context.
"""
)

# Sidebar
with st.sidebar:
    st.title("RAG Assistant")
    st.markdown("Use this assistant to query document content intelligently.")
    st.markdown("---")
    if st.button(" Clear Chat"):
        st.session_state.messages = [system_message]
        st.rerun()

# Initialize LLM (Groq)
if "llm" not in st.session_state:
    llm = ChatGroq(
        model=config["GROQ_API_MODEL"],
        api_key=config["GROQ_API_KEY"],
        temperature=0.1,
        max_tokens=131072,
    )
    st.session_state.llm = llm
else:
    llm = st.session_state.llm

# Title
st.title("RAG System with Groq LLM")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [system_message]

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    with st.chat_message(
        "user" if isinstance(message, HumanMessage) else "assistant"
                         ):
        st.write(message.content)

#  Generate response using RAG


def give_response(msg: str):
    # Step 1: Query Chroma for similar chunks
    results = collection.query(
        query_texts=[msg],
        n_results=3
    )

    # Step 2: Extract top documents
    docs = results.get("documents", [[]])[0]
    context = "\n\n".join(docs)

    # Step 3: Add context to system message
    context_message = SystemMessage(
        content=f"Context:\n{context}"
    )

    # Step 4: Build prompt for LLM
    messages = [system_message, context_message] + st.session_state.messages

    # Step 5: Get response from LLM
    response = llm.invoke(messages)

    # Step 6: Save AI response
    st.session_state.messages.append(AIMessage(content=response.content))

    return response

# Chat input


if msg := st.chat_input("Enter your question here:"):
    st.session_state.messages.append(HumanMessage(content=msg))
    with st.spinner("Thinking..."):
        response = give_response(msg)
    st.rerun()
