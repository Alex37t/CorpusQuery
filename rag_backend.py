import os
import time
import shutil
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import MultiQueryRetriever

# --- Configuration Constants ---
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.5-pro"
DOCS_DIRECTORY = 'docs'
VECTOR_DB_DIRECTORY = 'chroma_db_rag'
# Batched embedding in case of API rate limits
BATCH_SIZE = 25
SECONDS_BETWEEN_REQUESTS = 1


def get_api_key():
    """
    Retrieve the Google API key, checking Streamlit secrets first, then environment variables.
    """
    try:
        import streamlit as st
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if api_key:
            return api_key
    except ImportError:
        # Streamlit is not installed or not running
        pass
    except FileNotFoundError:
        # secrets.toml doesn't exist
        pass

    # Get environment variable for CLI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Error: GOOGLE_API_KEY not found. "
            "For Streamlit, add it to .streamlit/secrets.toml. "
            "For CLI, set it as an environment variable."
        )
    return api_key


def load_and_chunk_documents():
    """
    Loads documents from the DOCS_DIRECTORY, supporting both PDF and EPUB formats.
    It then splits the loaded documents into manageable chunks.
    Returns:
        list: A list of document chunks.
    Raises:
        FileNotFoundError: If no supported documents are found.
    """
    print(f"-> Loading documents from '{DOCS_DIRECTORY}'...")

    loaders = []
    for filename in os.listdir(DOCS_DIRECTORY):
        file_path = os.path.join(DOCS_DIRECTORY, filename)
        if filename.endswith('.pdf'):
            loaders.append(PyPDFLoader(file_path))
        elif filename.endswith('.epub'):
            loaders.append(UnstructuredEPubLoader(file_path))

    if not loaders:
        print(f"Warning: No PDF or EPUB files found in '{DOCS_DIRECTORY}'. The database will be empty.")
        return []

    # Load all documents from the prepared loaders
    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {loader.file_path}: {e}")

    print(f"-> Loaded {len(documents)} document pages/sections.")

    # Split documents into smaller chunks
    print("-> Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"-> Created {len(chunks)} text chunks.")
    return chunks


def create_vector_store(chunks):
    """
    Creates the vector store by embedding the document chunks in batches.
    Args:
        chunks (list): A list of document chunks to embed.

    Returns:
        Chroma: The created vector store instance.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    print("Creating new vector store with batched embedding...")
    vector_store = None
    # Calculate the total number of batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total batches: {total_batches}")
    # Loop through the chunks in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        # Get the current batch of chunks
        batch_chunks = chunks[i:i + BATCH_SIZE]
        # First batch creates the Chroma database.
        if i == 0:
            vector_store = Chroma.from_documents(
                documents=batch_chunks,
                embedding=embeddings,
                persist_directory=VECTOR_DB_DIRECTORY
            )
        else:
            # Subsequent batches add to the existing database
            vector_store.add_documents(documents=batch_chunks)

        # Progress updates
        current_batch_num = (i // BATCH_SIZE) + 1
        print(f"Processed batch {current_batch_num}/{total_batches}")

        # Delay between batches to respect API rate limits
        if current_batch_num < total_batches:
            time.sleep(SECONDS_BETWEEN_REQUESTS)

    print(f"Successfully saved {len(chunks)} chunks to {VECTOR_DB_DIRECTORY}.")
    return vector_store


def get_or_create_vector_store(rebuild=False):
    """
    Manages the vector store. Loads an existing one or creates a new one.

    Args:
        rebuild (bool): If True, forces the deletion and recreation of the DB.

    Returns:
        Chroma: The loaded or created vector store instance.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    if os.path.exists(VECTOR_DB_DIRECTORY) and not rebuild:
        print(f"-> Loading existing vector store from '{VECTOR_DB_DIRECTORY}'...")
        return Chroma(
            persist_directory=VECTOR_DB_DIRECTORY, embedding_function=embeddings
        )

    print("-> Proceeding to build/rebuild the vector database.")
    if os.path.exists(VECTOR_DB_DIRECTORY):
        print(f"--> Removing old database at '{VECTOR_DB_DIRECTORY}'...")
        try:
            shutil.rmtree(VECTOR_DB_DIRECTORY)
            print(f"Successfully deleted old database at '{VECTOR_DB_DIRECTORY}'.")
        except OSError as e:
            print(f"Error: {e.filename}, {e.strerror}")

    chunks = load_and_chunk_documents()
    if not chunks:
        raise ValueError("No documents were loaded to process.")

    return create_vector_store(chunks)


def create_rag_chain(vector_store, k=8, lambda_mult=0.5):
    """
    Creates the main RAG chain with MMR

    Args:
        vector_store: The Chroma vector store instance.
        k (int): The number of text chunks to retrieve for context (default: 8).
        lambda_mult (float): The diversity factor for MMR (0 for max diversity, 1 for max relevance).
    """

    # LLM for generating the final answer
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.1)
    # Deterministic LLM to generate search queries
    retriever_llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0)

    # MMR Retriever
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "lambda_mult": lambda_mult,
            "fetch_k": 50
        }
    )
    # Wrap the MMR retriever in a MultiQueryRetriever
    final_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=retriever_llm
    )

    prompt_path = Path("prompts/prompt.md")
    with prompt_path.open("r") as f:
        prompt_template_str = f.read()
    prompt = ChatPromptTemplate.from_template(prompt_template_str)

    def format_docs(docs):
        """Helper function to join all retrieved page_content into a single string."""
        formatted_context = "\n\n".join(doc.page_content for doc in docs)

        return formatted_context

    # LCEL chain
    rag_chain_with_sources = RunnableParallel(
        {"context": final_retriever, "question": RunnablePassthrough()}
    ).assign(
        answer=(
                RunnableParallel({
                    "context": lambda x: format_docs(x["context"]),
                    "question": RunnablePassthrough()
                })
                | prompt
                | llm
                | StrOutputParser()
        )
    )

    return rag_chain_with_sources
