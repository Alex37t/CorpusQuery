# app.py

"""
Streamlit web interface for CorpusQuery
"""

import os
import streamlit as st
import rag_backend as backend

st.set_page_config(
    page_title="CorpusQuery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 CorpusQuery")
st.caption("Query and explore your personal corpus of PDF and EPUB documents.")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Search Parameters")

    k_slider = st.number_input(
        "Chunks to Retrieve (k)",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
        help="Number of text chunks retrieved to form the context. Default = 8",
    )

    lambda_slider = st.slider(
        "Result Diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Controls the diversity of the search results. 0.0 = maximum diversity, 1.0 = maximum relevance. Default = 0.5"
    )

    st.header("Knowledge Base Management")
    st.info(
        """
        To rebuild the knowledge base with new or changed documents:

        1.  **Stop this Streamlit app.**
        2.  **Delete the `chroma_db_rag` folder** in your project directory.
        3.  **Restart the app** with `streamlit run app.py`

        The database will be created automatically on startup if it doesn't exist.
        """
    )


@st.cache_resource
def load_rag_chain(k, lambda_mult):
    """Loads the RAG chain with the specified MMR parameters."""
    try:
        vector_store = backend.get_or_create_vector_store()
        return backend.create_rag_chain(vector_store, k, lambda_mult)
    except Exception as e:
        st.error(f"An error occurred during initialization: {e}", icon="🚨")
        st.stop()


# Load the RAG chain on app start.
with st.spinner("Loading your corpus... Please wait."):
    rag_chain = load_rag_chain(k_slider, lambda_slider)

# --- Chat Interface ---
# Initialize chat history.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to CorpusQuery. Ready to query your document corpus."}
    ]

# Display chat history on each rerun.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input.
if prompt := st.chat_input("Ask a question about your corpus..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response.
    with st.chat_message("assistant"):
        with st.spinner("Querying corpus..."):
            response = rag_chain.invoke(prompt)
            answer = response["answer"]

            # Format and display sources.
            sources = "SOURCES:\n"
            for i, doc in enumerate(response["context"]):
                source_file = doc.metadata.get('source', 'Unknown')
                page_number = doc.metadata.get('page')
                category = doc.metadata.get('category')
                subject = doc.metadata.get('subject')

                location_info = ""
                if page_number is not None:
                    location_info = f"Page: {page_number + 1}"
                elif category:
                    location_info = f"Section: {category}"
                elif subject:
                    location_info = f"Section: {subject}"
                else:
                    location_info = "Location N/A"

                sources += f"- {i + 1}: {os.path.basename(source_file)} ({location_info})\n"

            full_response = f"{answer}\n\n---\n{sources}"
            st.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
