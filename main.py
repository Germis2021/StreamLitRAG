import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader


# === Aplinka ir API rakto įkėlimas ===
load_dotenv()
token = os.getenv("SECRET")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# === Įkeliam dokumentus iš įvairių šaltinių ===
loader = WebBaseLoader(
    web_paths=[
        "https://lt.wikipedia.org/wiki/Klaipėda",
        "https://lt.wikipedia.org/wiki/Klaipėdos_istorija",
        "https://klaipeda.lt",
    ]
)
docs = loader.load()

# === Įkeliame papildomai Klaipeda.txt ===
txt_loader = TextLoader("Klaipeda.txt", encoding="utf-8")
txt_docs = txt_loader.load()
for doc in txt_docs:
    doc.metadata["source"] = "Klaipeda.txt"

# === Sujungiame visus dokumentus ===
docs += txt_docs



# === Įsitikriname, kad kiekvienas dokumentas turi 'source' ===
for doc in docs:
    if "source" not in doc.metadata:
        doc.metadata["source"] = doc.metadata.get("url", "Nežinomas šaltinis")

# === Suskaidome į chunk'us ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# === Sukuriame embeddings ir vektorinę atmintį ===
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
)

vectorstore = InMemoryVectorStore(embeddings)
_ = vectorstore.add_documents(documents=splits)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === Promptas ir LLM ===
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(
    base_url=endpoint,
    api_key=token,
    model=model,
    temperature=1.0
)

# === Funkcija kontekstui suformuoti iš chunk'ų ===
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# === Funkcija chunk'ų šaltiniams atvaizduoti ===
def get_sources(docs):
    sources = []
    for doc in docs:
        source = doc.metadata.get('source', 'Nežinomas šaltinis')
        snippet = doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else "")
        sources.append((source, snippet))
    return sources

# === Rag grandinė ===
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# === Streamlit sąsaja ===
st.set_page_config(page_title="Klaipėda Chatbot", layout="centered")
st.title(" Klaipėdos Chatbotas")

# Pokalbio istorija
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Vartotojo klausimas
user_input = st.chat_input("Paklauskite apie Klaipėdą...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Ieškoma atsakymo..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Parodome naudotus chunk'us su šaltiniais
    relevant_docs = retriever.invoke(user_input)
    sources = get_sources(relevant_docs)
    with st.expander("📚 Naudoti teksto fragmentai (chunk'ai) ir šaltiniai:"):
        for i, (source, snippet) in enumerate(sources, 1):
            st.markdown(f"**{i}. Šaltinis: {source}**")
            st.write(snippet)
