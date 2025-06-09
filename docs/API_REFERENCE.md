# API Reference

This document provides detailed API documentation for the RAG chatbot components.

## DocumentLoader

### Class: `DocumentLoader`

Handles loading and processing of documents from the data directory.

#### Constructor
```python
DocumentLoader(data_dir: str = "data", chunk_size: int = 1000, chunk_overlap: int = 200)
```

**Parameters:**
- `data_dir` (str): Directory containing source documents
- `chunk_size` (int): Size of text chunks in characters
- `chunk_overlap` (int): Overlap between chunks in characters

#### Methods

##### `load_documents() -> List[Document]`
Load all supported documents from the data directory.

**Returns:** List of loaded and chunked documents

##### `get_document_stats() -> dict`
Get statistics about the documents in the data directory.

**Returns:** Dictionary with document statistics

## FAISSEmbedder

### Class: `FAISSEmbedder`

Handles embedding generation and FAISS vector storage.

#### Constructor
```python
FAISSEmbedder(model_name: str = "text-embedding-ada-002", 
              index_path: str = "outputs/faiss_index",
              documents_path: str = "outputs/documents.pkl")
```

**Parameters:**
- `model_name` (str): OpenAI embedding model to use
- `index_path` (str): Path to save/load FAISS index
- `documents_path` (str): Path to save/load document metadata

#### Methods

##### `create_index(documents: List[Document]) -> None`
Create a FAISS index from a list of documents.

**Parameters:**
- `documents` (List[Document]): List of documents to embed and index

##### `save_index() -> None`
Save the FAISS index and documents to disk.

##### `load_index() -> bool`
Load the FAISS index and documents from disk.

**Returns:** True if loading was successful

##### `similarity_search(query: str, k: int = 4, score_threshold: float = 0.7) -> List[Tuple[Document, float]]`
Perform similarity search for a query.

**Parameters:**
- `query` (str): The search query
- `k` (int): Number of results to return
- `score_threshold` (float): Minimum similarity score threshold

**Returns:** List of (document, score) tuples

##### `get_index_stats() -> dict`
Get statistics about the current FAISS index.

**Returns:** Dictionary with index statistics

## RAGChain

### Class: `RAGChain`

Constructs and manages the Retrieval-Augmented Generation chain.

#### Constructor
```python
RAGChain(model_name: str = "gpt-3.5-turbo",
         temperature: float = 0.1,
         max_tokens: int = 1000,
         retriever=None)
```

**Parameters:**
- `model_name` (str): OpenAI model to use for generation
- `temperature` (float): Temperature for response generation
- `max_tokens` (int): Maximum tokens in response
- `retriever`: Document retriever (FAISS-based)

#### Methods

##### `query(question: str) -> Dict[str, Any]`
Query the RAG system with a question.

**Parameters:**
- `question` (str): The question to ask

**Returns:** Response containing answer and source documents

##### `set_retriever(retriever) -> None`
Set the document retriever for the RAG chain.

**Parameters:**
- `retriever`: Document retriever (e.g., FAISS-based)

## RAGChatbot

### Class: `RAGChatbot`

Main RAG Chatbot class that orchestrates all components.

#### Constructor
```python
RAGChatbot(data_dir: str = "data",
           model_name: str = "gpt-3.5-turbo",
           embedding_model: str = "text-embedding-ada-002",
           temperature: float = 0.1,
           max_tokens: int = 1000)
```

**Parameters:**
- `data_dir` (str): Directory containing source documents
- `model_name` (str): OpenAI model for generation
- `embedding_model` (str): OpenAI model for embeddings
- `temperature` (float): Temperature for generation
- `max_tokens` (int): Maximum tokens in response

#### Methods

##### `initialize(force_rebuild: bool = False) -> bool`
Initialize the RAG system by loading documents and setting up the index.

**Parameters:**
- `force_rebuild` (bool): Force rebuild the index even if it exists

**Returns:** True if initialization was successful

##### `ask_question(question: str) -> Dict[str, Any]`
Ask a question and get a response.

**Parameters:**
- `question` (str): The question to ask

**Returns:** Response with answer and sources

##### `get_system_stats() -> Dict[str, Any]`
Get comprehensive system statistics.

**Returns:** Dictionary with system statistics

##### `rebuild_index() -> bool`
Rebuild the FAISS index from scratch.

**Returns:** True if rebuild was successful

## Response Format

All query responses follow this structure:

```python
{
    "question": str,           # Original question
    "answer": str,             # Generated answer
    "source_documents": List,  # List of source documents used
    "timestamp": str,          # ISO timestamp
    "model": str,              # Model used for generation
    "error": bool              # True if an error occurred (optional)
}
```

## Error Handling

All components include comprehensive error handling:

- **Document Loading Errors**: Logged and skipped, allowing partial processing
- **Embedding Errors**: Retried with exponential backoff
- **API Errors**: Graceful degradation with user-friendly error messages
- **Index Errors**: Automatic fallback to rebuild mode

## Configuration

### Environment Variables

```env
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL_NAME=gpt-3.5-turbo
```

### Logging

All components use Python's standard logging module:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL 