
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = "gpt-4o-mini"  # or "gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Replace this with your open ai key "SK-"


# Constants
DATA_FILE = "data/onboarding.txt"
CHROMA_DIR = "chroma_store"
EMBED_MODEL = "text-embedding-3-small"


# Load and embed text
@st.cache_resource
def load_vectorstore():
    loader = TextLoader(DATA_FILE)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL,openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb

# Streamlit UI
st.title("📚 RAG Chatbot with Your Text File")
st.markdown("Ask anything based on the provided text document.")

query = st.text_input("Ask a question:", placeholder="e.g., What is our onboarding process?")

if query:
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0,openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
    st.markdown("**Answer:**")
    st.write(response)
