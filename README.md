# Langchain Multi-File RAG System

This project implements a Retrieval Augmented Generation (RAG) system using Langchain. It allows you to process various types of files (PDF, TXT, DOCX, XLSX, HTML, JSON, Images), store their content in individual FAISS vector stores, and then ask questions against specific files, utilizing a hybrid search mechanism.

## Features

- Supports multiple file types: PDF, TXT, DOCX, XLSX, HTML, JSON, PNG, JPG, JPEG.
- Creates a separate FAISS vector index for each processed file.
- Uses OpenAI embeddings for document chunks.
- Employs a hybrid search strategy (semantic + keyword) for retrieval.
- Generates answers using an OpenAI language model (e.g., GPT-3.5 Turbo).
- Command-line interface (`app.py`) for processing files and asking questions.

## Prerequisites

- Python 3.8+
- An OpenAI API Key

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up OpenAI API Key:**
    You need to set your OpenAI API key as an environment variable.
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    On Windows, you can use:
    ```bash
    set OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Alternatively, you can set it in your system's environment variables. This key is required for generating embeddings and for the language model to answer questions.

## Directory Structure

```
.
├── app.py                  # Main CLI application
├── file_processing.py      # Handles file loading, chunking, FAISS creation
├── retrieval.py            # Handles FAISS loading, hybrid search, answer generation
├── requirements.txt        # Python dependencies
├── faiss_indexes/          # Directory created to store FAISS indexes (one per file)
└── README.md               # This file
```

## Usage

The primary interface is `app.py`. It has two main commands: `process` and `ask`.

**1. Processing Files (`process`)**

This command loads the content of the specified file(s), chunks the text, generates embeddings, and stores them in a FAISS index specific to each file. These indexes are saved in the `faiss_indexes/` directory.

*   **Process a single file:**
    ```bash
    python app.py process /path/to/your/document.pdf
    ```
    Replace `/path/to/your/document.pdf` with the actual path to your file.

*   **Process all supported files in a directory:**
    ```bash
    python app.py process /path/to/your/documents_directory/
    ```
    This will scan the directory for files with supported extensions (`.pdf`, `.txt`, `.docx`, `.xlsx`, `.html`, `.json`, `.png`, `.jpg`, `.jpeg`) and process each one.

**Important Note on Processing:** Processing involves calls to the OpenAI API for embeddings, which may incur costs depending on your OpenAI usage and plan.

**2. Asking Questions (`ask`)**

After a file has been processed and its index created, you can ask questions about it.

*   **Ask a question about a specific file:**
    ```bash
    python app.py ask --file /path/to/your/document.pdf --query "What is the main topic of this document?"
    ```
    - Replace `/path/to/your/document.pdf` with the path to the file you processed.
    - Replace `"What is the main topic of this document?"` with your actual query.

**Important Note on Asking:** Asking questions also involves calls to the OpenAI API (for the FAISS index loading if embeddings are re-verified, and for the language model to generate the answer), which may incur costs.

## How it Works

1.  **File Loading & Chunking (`file_processing.py`):**
    - `load_documents()`: Detects file type and uses appropriate Langchain loaders.
    - `chunk_documents()`: Splits loaded documents into smaller chunks using `RecursiveCharacterTextSplitter`.

2.  **Embedding & FAISS Storage (`file_processing.py`):**
    - `create_and_store_faiss_index()`:
        - Generates embeddings for chunks using `OpenAIEmbeddings`.
        - Creates a `FAISS` vector store from these embeddings.
        - Saves the index locally (e.g., `faiss_indexes/document_name.faiss` and `faiss_indexes/document_name.pkl`).

3.  **Retrieval & Answering (`retrieval.py`):**
    - `load_faiss_index()`: Loads a saved FAISS index for a given file.
    - `hybrid_search()`:
        - Performs semantic search using the FAISS index.
        - Performs keyword search on the original document chunks.
        - Combines and de-duplicates results.
    - `get_answer_from_documents()`:
        - Orchestrates loading the index and documents.
        - Uses the retrieved chunks from hybrid search as context.
        - Employs a `ChatOpenAI` model and a prompt template to generate an answer (RAG).

## Customization

-   **Embedding Model:** Currently uses `text-embedding-ada-002`. This can be changed in `file_processing.py` and `retrieval.py`.
-   **LLM Model:** Currently uses `gpt-3.5-turbo`. This can be changed in `retrieval.py`.
-   **Chunking Strategy:** Modify `chunk_size` and `chunk_overlap` in `file_processing.py`.
-   **Supported File Loaders:** Extend `loader_map` in `file_processing.py` to add more file types if compatible loaders are available.
