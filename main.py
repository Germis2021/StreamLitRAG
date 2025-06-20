from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

import os
from langchain import hub

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import bs4 

load_dotenv()  



token = os.getenv("SECRET")  
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"


loader = WebBaseLoader(
    web_paths=[
        "https://lt.wikipedia.org/wiki/Klaipėda",
        "https://lt.wikipedia.org/wiki/Klaipėdos_istorija",
        "https://klaipeda.lt",
        
    ],
    
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token, # type: ignore
)

vectorstore = InMemoryVectorStore(embeddings)

_ = vectorstore.add_documents(documents=splits)

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    
    return "\n\n".join(doc.page_content for doc in docs)

st.title("Streamlit LangChain Demo")

llm = ChatOpenAI(
    base_url="https://models.github.ai/inference",
    api_key=token,
    model="openai/gpt-4.1-nano",
    temperature=0.7
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


st.set_page_config(page_title="Klaipėda Chatbot", layout="centered")
st.title(" Klaipėdos Chatbotas")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Klausk apie Klaipėdą...")
if user_input:
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    
    with st.chat_message("assistant"):
        with st.spinner("Ieškoma atsakymo..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})