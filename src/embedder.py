"""
Embedder Module

This module handles the generation of embeddings for document chunks and their storage
in a FAISS vector database. It provides functionality for:
- Generating embeddings using OpenAI's embedding model
- Storing embeddings in FAISS for efficient similarity search
- Loading and saving FAISS indices
- Performing similarity searches
"""

import os
import logging
import pickle
from typing import List, Tuple, Optional
from pathlib import Path

import faiss
import numpy as np
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSEmbedder:
    """
    A class to handle embedding generation and FAISS vector storage for RAG systems.
    
    This class manages the conversion of text documents to embeddings and their
    storage in a FAISS index for efficient similarity search.
    """
    
    def __init__(self, 
                 model_name: str = "text-embedding-ada-002",
                 index_path: str = "outputs/faiss_index",
                 documents_path: str = "outputs/documents.pkl"):
        """
        Initialize the FAISS Embedder.
        
        Args:
            model_name (str): OpenAI embedding model to use
            index_path (str): Path to save/load FAISS index
            documents_path (str): Path to save/load document metadata
        """
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.documents_path = Path(documents_path)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize FAISS index and documents
        self.index = None
        self.documents = []
        self.dimension = None
        
    def create_index(self, documents: List[Document]) -> None:
        """
        Create a FAISS index from a list of documents.
        
        Args:
            documents (List[Document]): List of documents to embed and index
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return
            
        logger.info(f"Creating FAISS index for {len(documents)} documents")
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        # Get embedding dimension
        self.dimension = embeddings_array.shape[1]
        
        # Create FAISS index
        logger.info(f"Creating FAISS index with dimension {self.dimension}")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add embeddings to index
        self.index.add(embeddings_array)
        
        # Store documents
        self.documents = documents
        
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
        
    def save_index(self) -> None:
        """
        Save the FAISS index and documents to disk.
        """
        if self.index is None:
            logger.warning("No index to save")
            return
            
        # Create output directory
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"FAISS index saved to {self.index_path}")
        
        # Save documents
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Documents saved to {self.documents_path}")
        
    def load_index(self) -> bool:
        """
        Load the FAISS index and documents from disk.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            if not self.index_path.exists() or not self.documents_path.exists():
                logger.warning("Index files not found")
                return False
                
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"FAISS index loaded with {self.index.ntotal} vectors")
            
            # Load documents
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            logger.info(f"Documents loaded: {len(self.documents)}")
            
            # Get dimension from loaded index
            self.dimension = self.index.d
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
            
    def similarity_search(self, 
                         query: str, 
                         k: int = 4,
                         score_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """
        Perform similarity search for a query.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            score_threshold (float): Minimum similarity score threshold
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        if self.index is None:
            logger.error("No index available for search")
            return []
            
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize query vector
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, k)
        
        # Filter results and create document-score pairs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= score_threshold:  # -1 indicates no result
                results.append((self.documents[idx], float(score)))
                
        logger.info(f"Found {len(results)} relevant documents for query: '{query}'")
        return results
        
    def get_index_stats(self) -> dict:
        """
        Get statistics about the current FAISS index.
        
        Returns:
            dict: Statistics about the index
        """
        if self.index is None:
            return {"total_vectors": 0, "dimension": 0, "is_trained": False}
            
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "is_trained": self.index.is_trained
        }
        
    def rebuild_index(self, documents: List[Document]) -> None:
        """
        Rebuild the FAISS index with new documents.
        
        Args:
            documents (List[Document]): New documents to index
        """
        logger.info("Rebuilding FAISS index")
        self.create_index(documents)
        self.save_index()
        
    def delete_index(self) -> None:
        """
        Delete the saved index files.
        """
        try:
            if self.index_path.exists():
                self.index_path.unlink()
                logger.info(f"Deleted FAISS index: {self.index_path}")
                
            if self.documents_path.exists():
                self.documents_path.unlink()
                logger.info(f"Deleted documents: {self.documents_path}")
                
        except Exception as e:
            logger.error(f"Error deleting index files: {str(e)}")


class EmbeddingManager:
    """
    A high-level manager for embedding operations.
    
    This class provides a simplified interface for common embedding tasks
    and manages the lifecycle of the FAISS embedder.
    """
    
    def __init__(self, 
                 model_name: str = "text-embedding-ada-002",
                 index_path: str = "outputs/faiss_index",
                 documents_path: str = "outputs/documents.pkl"):
        """
        Initialize the Embedding Manager.
        
        Args:
            model_name (str): OpenAI embedding model to use
            index_path (str): Path to save/load FAISS index
            documents_path (str): Path to save/load document metadata
        """
        self.embedder = FAISSEmbedder(
            model_name=model_name,
            index_path=index_path,
            documents_path=documents_path
        )
        
    def setup_index(self, documents: List[Document], force_rebuild: bool = False) -> bool:
        """
        Set up the FAISS index, either by loading existing or creating new.
        
        Args:
            documents (List[Document]): Documents to index if creating new
            force_rebuild (bool): Force rebuild even if index exists
            
        Returns:
            bool: True if setup was successful
        """
        if not force_rebuild and self.embedder.load_index():
            logger.info("Loaded existing FAISS index")
            return True
        else:
            logger.info("Creating new FAISS index")
            self.embedder.create_index(documents)
            self.embedder.save_index()
            return True
            
    def search(self, query: str, k: int = 4, score_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """
        Perform similarity search.
        
        Args:
            query (str): Search query
            k (int): Number of results
            score_threshold (float): Minimum score threshold
            
        Returns:
            List[Tuple[Document, float]]: Search results
        """
        return self.embedder.similarity_search(query, k, score_threshold)
        
    def get_stats(self) -> dict:
        """
        Get index statistics.
        
        Returns:
            dict: Index statistics
        """
        return self.embedder.get_index_stats()


if __name__ == "__main__":
    # Test the embedder
    from loaders import DocumentLoader, create_sample_documents
    
    # Create sample documents
    create_sample_documents()
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    # Create embedder
    embedder = FAISSEmbedder()
    embedder.create_index(documents)
    embedder.save_index()
    
    # Test search
    results = embedder.similarity_search("What is machine learning?")
    for doc, score in results:
        print(f"Score: {score:.3f}")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("-" * 50) 