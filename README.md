# RETRIEVIAL ARGUMENTED GENERATION AI SYSTEM

A Retrieval Augmented Generation system is an advanced architecture that combines retrieval based and generative methods in natural language processing (NLP). It’s primarily used to answer user queries or generate text by leveraging both large language models (LLMs) and external data sources.

# WHAT IS EXACTLY IN MY PROJECT

Basing on my project, A literture student can gain access to quick answers from the novel named Romeo and Juliet. That means a student or user can can prompt that RAG system and ask a specific question basing on the context provide to the system and he or she can get quick access to there answers.

# TECHNOLOGY USED

In this project I used langchain that is a framework used by dvelopers to build applications powered by large language model like Openai, GPT models GROQ AI among others. so during the process I installed using the command "python -m pip langchain-groq" because my project uses groq AI.

On using a llama model that provided me with less tokens yet the uploaded document produced more tokens than the model could support, to solve this issue I decided to chunk the text and I used chunking_evaluation strategy called ClusterSemanticChunker after which I created embeddings using OpenAIEmbeddings and then stored them in a vector database, and for this I decided to use chromadb

# Dockerization 

I decided to dockerize the streamlit app in a docker file and since the chromadb image already existed, I just created a docker-compose.yaml where I connected the the streamlit app to the chromadb that contained the embeddings creatied