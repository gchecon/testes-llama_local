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
import subprocess

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
    """
    Check if a file identified by its hash already exists in the vectorstore.

    This function interacts with a SQLite database to determine whether the given
    `file_hash` exists within the specified `db_path`. It performs a query on the
    `embedding_metadata` table, specifically checking for the key `file_hash` with
    a matching value. The function returns a boolean indicating the presence of
    the file hash in the database.

    :param file_hash: The hash of the file to be checked.
    :type file_hash: str
    :param db_path: The SQLite database path containing the table `embedding_metadata`.
    :type db_path: str
    :return: True if the file hash exists in the database, False otherwise.
    :rtype: bool
    :raises sqlite3.Error: If there is an error accessing the database.
    """
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

# def get_available_models():
#     try:
#         response = requests.get(f"{OLLAMA_HOST}/models")
#         if response.status_code == 200:
#             return response.json()
#         else:
#             st.error(f"Erro ao obter modelos do Ollama: {response.status_code}")
#             return []
#     except requests.exceptions.RequestException as e:
#         st.error(f"Erro de conexão com Ollama: {e}")
#         return []

def get_ollama_models():
    """
    Executes a Docker command to list available Ollama models running in the
    Ollama server container. Processes the output of the command and extracts
    the model names, if present. The function handles both subprocess errors
    and unexpected exceptions, ensuring robustness in case of failures.

    :return: A list of strings representing model names extracted from the
        Ollama server's output. Returns an empty list in case of errors.
    :rtype: list[str]
    """
    try:
        # Executa o comando e captura a saída
        result = subprocess.run(
            ['docker', 'exec', 'ollama-server', 'ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        # A saída está em result.stdout
        # Vamos dividir por linhas e processar
        models = []
        for line in result.stdout.strip().split('\n'):
            # Pula a linha de cabeçalho se existir
            if 'NAME' in line or not line:
                continue
            # Divide a linha em colunas e pega o nome do modelo
            model_name = line.split()[0].split(':')[0]  # Pega apenas o que vem antes do ':'
            models.append(model_name)
        return models
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o comando: {e}")
        return []
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return []

def open_folder_dialog():
    """
    Prompts the user to select a folder from their filesystem using a graphical
    dialog window. This function initializes a hidden Tkinter root window, opens
    a folder selection dialog, and returns the path of the selected folder. The
    function does not interact with the terminal or command line.

    :raises TclError: If the Tkinter graphical environment fails to initialize
        or if the dialog cannot be created.

    :return: The absolute path of the folder selected by the user or an empty
        string if no folder is selected.
    :rtype: str
    """
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

def update_folder_history(folder):
    """
    Update the history of recently accessed folders. When a new folder is
    provided, it is added to the beginning of the historical list of folders
    maintained in the session state. If the list exceeds a maximum size of
    five folders, the oldest entry is removed to maintain the limit.

    :param folder: Path or name of the folder to add to the history.
    :type folder: str
    :return: None
    """
    if folder and folder not in st.session_state.last_folders:
        st.session_state.last_folders.insert(0, folder)
        if len(st.session_state.last_folders) > 5:
            st.session_state.last_folders.pop()

# Função para processar PDF e criar embeddings
def process_pdf(file_path, vectorstore):
    """
    Processes a PDF file by calculating its hash, checking if it is already present in the vectorstore,
    splitting its content into chunks, and finally adding these processed chunks to the vectorstore.

    The method ensures that the provided PDF is not reprocessed if it is already present in the vectorstore.
    If the PDF is new, it extracts its content, organizes it into manageable chunks, updates their metadata,
    and adds these chunks to a vector database.

    :param file_path: The path to the PDF file to be processed.
    :type file_path: str
    :param vectorstore: The vector store instance where the extracted chunks will be stored.
    :type vectorstore: Any
    :return: The number of chunks extracted and added to the vectorstore.
    :rtype: int
    """
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
    """
    Main script to manage the execution of a document-based conversational application.
    It orchestrates model selection, document processing, and chatbot interactions,
    providing mechanisms for users to select and configure available models and directories,
    load PDFs into memory, and engage in Q&A interaction with the processed content. The script
    handles error scenarios gracefully, ensures reactivity in the user interface, and uses session
    state to maintain continuity across interactions.

    Sections include settings configuration, interaction via sidebar and chat interface, and
    integrated document processing capabilities, utilizing language model configurations and
    database storage for embeddings.

    :raises Exception: Propagates exceptions raised while configuring models, processing PDFs,
                       or interacting with embeddings and retrievers.

    """
    # Obter modelos disponíveis
    if "available_models" not in st.session_state:
        st.session_state.available_models = get_ollama_models()

    if not st.session_state.available_models:
        st.error("Nenhum modelo foi encontrado.")

    with st.sidebar:

        st.title("Configurações")
        model = st.selectbox(
            "Selecione o Modelo LLM",
            st.session_state.available_models,
            index = 0 if st.session_state.available_models else None
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
                question_generator=llm, ## Verificar
                return_source_documents=True
            )

            # print(f"Histórico do chat: {st.session_state.memory.chat_memory}")

            # Invocação da cadeia
            response = retrieval_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.memory.chat_memory or []
            })

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
