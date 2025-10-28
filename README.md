# CorpusQuery

CorpusQuery is a conversational RAG application built with Langchain that transforms your personal library of
documents (PDFs, EPUBs) into a knowledge engine that answers your queries.

## Features

* **Conversational Q&A:**  Chat interface for all document queries.
* **Multi-Format Support:** Ingests both PDF and EPUB files into a unified knowledge base.
* **Retrieval Pipeline:** Uses `MultiQueryRetriever` and `Maximal Marginal Relevance (MMR)` search to provide context
  that is both relevant and diverse.
* **Tunable Retrieval:** Adjust retrieval parameters in real time.
* **Batch Ingestion:**  Embed documents with adjustable batch sizes in case of certain API rate limits.
* **GUI + CLI:**  Web interface or command line access.

## Setup & Installation

**1. Install Pandoc (Required for EPUB support)**

* Download the official Pandoc installer for your operating system from the **[ releases page](https://github.com/jgm/pandoc/releases)**.
* Run the installer to complete the setup.

**2. Clone the Repository & Install Python Dependencies**

```bash
git clone https://github.com/Alex37t/CorpusQuery.git
cd CorpusQuery

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install the required Python packages
pip install -r requirements.txt
```

**3. Configure Your Google API Key**

* **Get a free or paid API key:** Visit **[Google AI Studio](https://aistudio.google.com/app/apikey)** and create your
  key.
* **Configure for the GUI use (Streamlit):**

  Copy the example secrets file and edit it with your API key:
  ```bash
  # For macOS/Linux
  cp .streamlit/secrets.toml.example .streamlit/secrets.toml

  # For Windows
  copy .streamlit\secrets.toml.example .streamlit\secrets.toml
  ```
  Now, edit the ".streamlit/secrets.tom" file and replace "your-api-key-here" with your actual key.


* **Configure for CLI use:**

  Set the key as an environment variable in your terminal:
  ```
  export GOOGLE_API_KEY="your-api-key-here"
  ```
    - On Windows: Add key to Environment Variables. See guide
      in [Gemini API docs](https://ai.google.dev/gemini-api/docs/api-key#windows)

**4. Add Your Documents**

Place your `.pdf` and `.epub` files inside the `docs/` directory.

## Usage

#### GUI

To launch the application web interface:

```
streamlit run app.py
```

#### CLI

To launch the application via the terminal:

  *Interactive Mode:*
  ```bash
  python cli.py
  ```
  *Direct Question:*
  ```bash
  python cli.py --question  "According to the author, what is the relationship between x and y?"
  ```

  *To ask a broad question requiring more context:*
  ```bash
  python cli.py --question "Summarize the key themes of x" --k 12
  ```

  *To ask a specific question where you want the most relevant, least diverse results:*
  ```bash
  python cli.py --question "What was the exact date of the event x?" --k 3 --lambda-mult 0.9
  ```

## Rebuilding the Database

The knowledge base is built automatically on first run. To rebuild the database:

**CLI:**

Use the `--rebuild-db` flag:

```bash
python cli.py --rebuild-db
```

**GUI:**

1. **Stop the Streamlit app** (`Ctrl+C`) in the terminal.
2. **Delete the `chroma_db_rag` folder** in your project directory.
3. **Restart the app** with `streamlit run app.py`.

## Configuration & Tuning

You can control the retrieval behavior via the GUI sliders on the web app or using CLI parameters.

## CLI Parameters

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--question` | string | (none) | The question you want to ask. If omitted, starts interactive mode. |
| `--rebuild-db`| flag | N/A | Force a rebuild of the vector database. |
| `--k` | integer| `8` | The number of text chunks to retrieve for context. |
| `--lambda-mult` | float | `0.5` | The MMR diversity factor. `0.0` for maximum diversity, `1.0` for maximum relevance. |

## LLM Configuration

Note: For now only Google's Gemini models are supported.

Constants are defined at the top of the `rag_backend.py` file and can be edited there directly
| Constant | Default Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME`| `models/gemini-embedding-001` | The model used to generate embeddings. |
| `LLM_MODEL_NAME` | `gemini-2.5-pro` | Main LLM model for generating answers. |
| `BATCH_SIZE` | `25` | Number of chunks to embed in a single API call during database creation. |
| `SECONDS_BETWEEN_REQUESTS` | `1.0` | Seconds to pause between database embedding requests. |

**Prompt:**

The system prompt that guides the LLM's behavior is at **`prompts/prompt.md`**. You can edit the file to change the
model's persona, tone, or how strictly it adheres to the retrieved context.
