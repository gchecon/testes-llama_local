# app.py
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

# Carrega variáveis de ambiente
load_dotenv()

# Obtém as configurações do arquivo .env
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', './chroma_db')
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'sentence-transformers/multilingual-mpnet-base-v2')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 120))

# Configurações do Streamlit
st.set_page_config(layout="wide")

# Inicializa as variáveis de estado da sessão
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()


# Função para calcular o hash do arquivo
def calculate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# Função para processar PDF e criar embeddings
def process_pdf(file_path, vectorstore):
    # Verifica se o arquivo já foi processado
    file_hash = calculate_file_hash(file_path)
    if file_hash in st.session_state.processed_files:
        st.warning(f"Arquivo {os.path.basename(file_path)} já foi processado anteriormente.")
        return

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Divide o texto em chunks usando valores do .env
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Adiciona metadados
    for chunk in chunks:
        chunk.metadata.update({
            "file_hash": file_hash,
            "file_name": os.path.basename(file_path),
            "processed_date": str(datetime.now())
        })

    # Adiciona ao vectorstore
    vectorstore.add_documents(chunks)

    # Registra o arquivo como processado
    st.session_state.processed_files.add(file_hash)

    return len(chunks)


# Interface principal
def main():
    # Sidebar para configurações
    with st.sidebar:
        st.title("Configurações")

        # Seleção de modelo
        model = st.selectbox(
            "Selecione o Modelo LLM",
            ["llama-3.2", "mistral", "neural-chat", "llama2:7b-chat"],
            index=0
        )

        # Upload de arquivos
        st.header("Processamento de Documentos")
        folder_path = st.text_input("Diretório dos PDFs")

        if folder_path:
            if os.path.exists(folder_path):
                if st.button("Carregar PDFs", disabled=False):
                    # Inicializa o vectorstore com modelo de embeddings do .env
                    embeddings = HuggingFaceEmbeddings(
                        model_name=EMBEDDINGS_MODEL
                    )

                    vectorstore = Chroma(
                        persist_directory=VECTOR_DB_PATH,
                        embedding_function=embeddings
                    )

                    # Processa os arquivos PDF no diretório
                    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
                    for pdf_file in pdf_files:
                        file_path = os.path.join(folder_path, pdf_file)
                        chunks_processed = process_pdf(file_path, vectorstore)
                        if chunks_processed:
                            st.success(f"Processado {pdf_file}: {chunks_processed} chunks")

                    vectorstore.persist()
            else:
                st.error("Diretório não encontrado")

    # Área principal
    st.title("Chat com Documentos")

    # Inicializa o chat se ainda não existir
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Campo de entrada do usuário
    if prompt := st.chat_input("Digite sua pergunta"):
        # Adiciona a pergunta do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepara o contexto e gera a resposta
        try:
            # Inicializa embeddings e vectorstore
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDINGS_MODEL
            )

            vectorstore = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )

            # Inicializa o modelo LLM com configurações do Ollama
            llm = Ollama(
                model=model,
                base_url=OLLAMA_HOST,
                timeout=OLLAMA_TIMEOUT
            )

            # Cria a chain de conversação
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=st.session_state.memory,
                return_source_documents=True
            )

            # Gera a resposta
            response = qa_chain({"question": prompt})
            answer = response['answer']

            # Adiciona a resposta ao histórico
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

        except Exception as e:
            st.error(f"Erro ao processar a pergunta: {str(e)}")


if __name__ == "__main__":
    main()