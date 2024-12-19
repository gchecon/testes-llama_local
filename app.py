import streamlit as st
import os
from dotenv import load_dotenv
import hashlib
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM  # Atualizado para a nova biblioteca Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
import requests
import tkinter as tk
from tkinter import filedialog
import sqlite3

# Carrega variáveis de ambiente
load_dotenv()

# Configurações iniciais
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', './chroma_db')
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'all-MiniLM-L6-v2')  # Modelo corrigido
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')  # Token da Hugging Face
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN

# Inicializa variáveis de estado
if 'last_folders' not in st.session_state:
    st.session_state.last_folders = []
if 'llm_instance' not in st.session_state:
    # Define a LLM default (Mistral via Ollama)
    st.session_state.llm_instance = OllamaLLM(model="mistral", base_url=OLLAMA_HOST)
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)  # Atualizado para persistir corretamente
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

def is_file_in_vectorstore(file_hash, db_path):
    try:
        # Conexão com o banco de dados SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Consulta para verificar se o file_hash já existe
        query = "SELECT COUNT(*) FROM embedding_metadata WHERE key = 'file_hash' AND string_value = ?"
        cursor.execute(query, (file_hash,))
        count = cursor.fetchone()[0]

        conn.close()
        return count > 0
    except sqlite3.Error as e:
        st.error(f"Erro ao acessar o banco de dados: {e}")
        return False

def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/models")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erro ao obter modelos do Ollama: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com Ollama: {e}")
        return []

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
    # Calcular o hash do arquivo
    file_hash = hashlib.sha256(open(file_path, "rb").read()).hexdigest()

    # Caminho do banco de dados
    db_path = os.path.join(VECTOR_DB_PATH, "chroma.sqlite3")

    # Verificar se o arquivo já está na base vetorial
    if is_file_in_vectorstore(file_hash, db_path):
        st.warning(f"Arquivo {os.path.basename(file_path)} já está na base vetorial.")
        return

    # Carregar e dividir o PDF em chunks
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                   length_function=len)
    chunks = text_splitter.split_documents(documents)

    # Atualizar os metadados dos chunks
    for chunk in chunks:
        chunk.metadata.update(
            {"file_hash": file_hash, "file_name": os.path.basename(file_path), "processed_date": str(datetime.now())}
        )

    # Adicionar os chunks ao vectorstore
    vectorstore.add_documents(chunks)
    st.success(f"Arquivo {os.path.basename(file_path)} processado e adicionado à base vetorial.")

    return len(chunks)

def main():
    # Obter modelos disponíveis
    if "available_models" not in st.session_state:
        st.session_state.available_models = get_available_models()

    # Garantir que Mistral seja o modelo padrão inicial
    default_model = "mistral"
    if default_model not in st.session_state.available_models:
        st.session_state.available_models.insert(0, default_model)  # Insere o Mistral se não estiver na lista

    with st.sidebar:
        st.title("Configurações")
        model = st.selectbox(
            "Selecione o Modelo LLM",
            st.session_state.available_models,
            index=st.session_state.available_models.index(default_model)
        )

        # Atualiza o modelo LLM na memória ao mudar a seleção
        if "llm_instance" not in st.session_state or model != st.session_state.llm_instance.model:
            try:
                st.session_state.llm_instance = OllamaLLM(model=model, base_url=OLLAMA_HOST)
                st.session_state.memory = ConversationBufferMemory(return_messages=True)
                st.success(f"Modelo configurado para: {model}")
            except Exception as e:
                st.error(f"Erro ao configurar o modelo {model}: {e}")

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
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
                vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
                pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
                for pdf_file in pdf_files:
                    file_path = os.path.join(folder_path, pdf_file)
                    chunks_processed = process_pdf(file_path, vectorstore)
                    if chunks_processed:
                        st.success(f"Processado {pdf_file}: {chunks_processed} chunks")

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
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
            vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
            llm = st.session_state.llm_instance

            # Configuração do combine_docs_chain
            combine_docs_chain = load_qa_chain(llm=st.session_state.llm_instance, chain_type="stuff")

            # Configuração da cadeia de recuperação
            retrieval_chain = ConversationalRetrievalChain(
                retriever=vectorstore.as_retriever(),
                combine_docs_chain=combine_docs_chain,
                return_source_documents=True
            )

            print(f"Histórico do chat: {st.session_state.memory.chat_memory}")

            # Invocação da cadeia
            response = retrieval_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.memory.chat_memory or []
            })

            print(f'Ponto 4 - Resposta: {response}')

            # Acessa o retorno de forma segura
            if isinstance(response, dict) and 'answer' in response:
                answer = response['answer']
            elif isinstance(response, tuple) or isinstance(response, list):
                answer = response[0]
            else:
                raise ValueError("Formato inesperado de resposta: {}".format(response))

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"Erro ao processar a pergunta: {str(e)}")

if __name__ == "__main__":
    main()
