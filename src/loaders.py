"""
Document Loaders Module

This module handles loading and processing of documents (PDF and TXT files)
from the data directory. It includes functionality for:
- Loading PDF files using PyPDF
- Loading text files
- Chunking documents using LangChain's RecursiveCharacterTextSplitter
- Document preprocessing and cleaning
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    A class to handle loading and processing of documents from the data directory.
    
    Supports PDF and TXT files, with automatic chunking for optimal RAG performance.
    """
    
    def __init__(self, data_dir: str = "data", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the DocumentLoader.
        
        Args:
            data_dir (str): Directory containing source documents
            chunk_size (int): Size of text chunks in characters
            chunk_overlap (int): Overlap between chunks in characters
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_documents(self) -> List[Document]:
        """
        Load all supported documents from the data directory.
        
        Returns:
            List[Document]: List of loaded and chunked documents
        """
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist. Creating it.")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return []
            
        documents = []
        supported_extensions = {'.pdf', '.txt'}
        
        for file_path in self.data_dir.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    file_docs = self._load_single_file(file_path)
                    documents.extend(file_docs)
                    logger.info(f"Loaded {len(file_docs)} chunks from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """
        Load a single file and return its document chunks.
        
        Args:
            file_path (Path): Path to the file to load
            
        Returns:
            List[Document]: List of document chunks
        """
        try:
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
                
            # Load raw documents
            raw_docs = loader.load()
            
            # Add metadata
            for doc in raw_docs:
                doc.metadata.update({
                    'source': str(file_path),
                    'filename': file_path.name,
                    'file_type': file_path.suffix.lower()
                })
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(raw_docs)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    def get_document_stats(self) -> dict:
        """
        Get statistics about the documents in the data directory.
        
        Returns:
            dict: Statistics about the documents
        """
        if not self.data_dir.exists():
            return {"total_files": 0, "file_types": {}}
            
        stats = {"total_files": 0, "file_types": {}}
        
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in {'.pdf', '.txt'}:
                stats["total_files"] += 1
                file_type = file_path.suffix.lower()
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
                
        return stats


def create_sample_documents():
    """
    Create sample documents in the data directory for testing.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample text document
    sample_text = """
    Artificial Intelligence and Machine Learning
    
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. Some of the activities computers with artificial intelligence are designed for include:
    
    - Speech recognition
    - Learning
    - Planning
    - Problem solving
    
    Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
    
    The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to analyze various factors of data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition.
    """
    
    with open(data_dir / "ai_ml_overview.txt", "w") as f:
        f.write(sample_text)
    
    # Sample document about RAG
    rag_text = """
    Retrieval-Augmented Generation (RAG)
    
    Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with external knowledge retrieval. This approach enhances the capabilities of language models by allowing them to access and reference specific information from external sources.
    
    Key Components of RAG:
    
    1. Document Loading: The process of importing documents from various sources (PDFs, text files, databases, etc.)
    
    2. Text Chunking: Breaking down large documents into smaller, manageable pieces that can be effectively processed and retrieved
    
    3. Embedding Generation: Converting text chunks into numerical vectors that capture semantic meaning
    
    4. Vector Storage: Storing embeddings in a vector database (like FAISS) for efficient similarity search
    
    5. Retrieval: Finding the most relevant document chunks based on user queries
    
    6. Generation: Using the retrieved context along with the user's question to generate accurate, grounded responses
    
    Benefits of RAG:
    - Provides up-to-date information
    - Reduces hallucinations
    - Improves accuracy with domain-specific knowledge
    - Enables traceability to source documents
    - Allows for easy knowledge updates
    
    Common Use Cases:
    - Customer support chatbots
    - Research assistants
    - Document Q&A systems
    - Knowledge management systems
    """
    
    with open(data_dir / "rag_system.txt", "w") as f:
        f.write(rag_text)
    
    logger.info("Sample documents created in data/ directory")


if __name__ == "__main__":
    # Test the document loader
    loader = DocumentLoader()
    create_sample_documents()
    docs = loader.load_documents()
    print(f"Loaded {len(docs)} document chunks")
    print(f"Document stats: {loader.get_document_stats()}") 