import streamlit as st
import os
from dotenv import load_dotenv
import hashlib
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
import json
import tkinter as tk
from tkinter import filedialog

# Carrega variáveis de ambiente
load_dotenv()

# Configurações iniciais
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', './chroma_db')
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'all-MiniLM-L6-v2')  # Modelo corrigido
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')  # Token da Hugging Face
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 120))

# Inicializa variáveis de estado
if 'last_folders' not in st.session_state:
    st.session_state.last_folders = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()


def open_folder_dialog():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected


def update_folder_history(folder):
    if folder and folder not in st.session_state.last_folders:
        st.session_state.last_folders.insert(0, folder)
        if len(st.session_state.last_folders) > 5:
            st.session_state.last_folders.pop()


# Função para processar PDF e criar embeddings
def process_pdf(file_path, vectorstore):
    file_hash = hashlib.sha256(open(file_path, "rb").read()).hexdigest()
    if file_hash in st.session_state.processed_files:
        st.warning(f"Arquivo {os.path.basename(file_path)} já foi processado.")
        return

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                   length_function=len)
    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata.update(
            {"file_hash": file_hash, "file_name": os.path.basename(file_path), "processed_date": str(datetime.now())})
    vectorstore.add_documents(chunks)
    st.session_state.processed_files.add(file_hash)
    return len(chunks)


def main():
    with st.sidebar:
        st.title("Configurações")
        model = st.selectbox("Selecione o Modelo LLM", ["mistral", "llama-3.2", "neural-chat", "llama2:7b-chat"],
                             index=0)
        st.header("Processamento de Documentos")

        # Combobox com histórico de pastas
        folder_path = st.selectbox("Diretório dos PDFs", options=[""] + st.session_state.last_folders)

        if st.button("Pesquisar"):
            new_folder = open_folder_dialog()
            if new_folder:
                update_folder_history(new_folder)
                st.session_state["selected_folder"] = new_folder
                st.rerun()

        # Exibe o diretório selecionado
        if "selected_folder" in st.session_state:
            folder_path = st.session_state["selected_folder"]
            st.text_input("Diretório Selecionado", value=folder_path, disabled=True)

        if folder_path:
            if st.button("Carregar PDFs"):
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL, token=HUGGINGFACE_TOKEN)
                vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
                pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
                for pdf_file in pdf_files:
                    file_path = os.path.join(folder_path, pdf_file)
                    chunks_processed = process_pdf(file_path, vectorstore)
                    if chunks_processed:
                        st.success(f"Processado {pdf_file}: {chunks_processed} chunks")
                vectorstore.persist()

    st.title("Chat com Documentos")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Digite sua pergunta"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL, token=HUGGINGFACE_TOKEN)
            vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
            llm = Ollama(model=model, base_url=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
            qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                             memory=st.session_state.memory)
            response = qa_chain({"question": prompt})
            answer = response['answer']
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"Erro ao processar a pergunta: {str(e)}")


if __name__ == "__main__":
    main()