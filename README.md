# RAG System for PDF Documents

A simple Retrieval-Augmented Generation (RAG) system that answers questions based on Ukrainian-language PDF documents.

## Technologies

- Python
- PyMuPDF — PDF text extraction
- LangChain — text chunking
- Sentence Transformers — embeddings
- NumPy — cosine similarity
- Ollama (Qwen3:4b) — answer generation

## How It Works

1. Extract text from PDF documents.
2. Split text into chunks.
3. Convert chunks into embeddings.
4. Find relevant chunks using cosine similarity.
5. Generate an answer using a local LLM.

## Installation

Install Python dependencies:

    pip install -r requirements.txt

Download the Ollama model:

    ollama pull qwen3:4b

## Usage

Place PDF files in the `data/` folder and run:

    python main.py

## Configuration

Edit the parameters in `main.py`:

- `CHUNK_SIZE` — maximum chunk size.
- `CHUNK_OVERLAP` — overlap between chunks.
- `TOP_K` — number of retrieved chunks.
- `QUESTION` — user question.

The project is designed to test how these parameters affect retrieval and answer quality.
