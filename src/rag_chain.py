"""
RAG Chain Module

This module constructs the Retrieval-Augmented Generation (RAG) chain using LangChain.
It combines document retrieval with language model generation to create a comprehensive
Q&A system. Features include:
- RetrievalQA chain construction
- Custom prompt templates
- Source document tracking
- Response formatting and validation
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChain:
    """
    A class to construct and manage the Retrieval-Augmented Generation chain.
    
    This class combines document retrieval with language model generation to create
    a comprehensive Q&A system that can answer questions based on indexed documents.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 retriever=None):
        """
        Initialize the RAG Chain.
        
        Args:
            model_name (str): OpenAI model to use for generation
            temperature (float): Temperature for response generation
            max_tokens (int): Maximum tokens in response
            retriever: Document retriever (FAISS-based)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retriever = retriever
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create the RAG chain
        self.chain = self._create_chain()
        
    def _create_chain(self) -> RetrievalQA:
        """
        Create the RetrievalQA chain with custom prompt template.
        
        Returns:
            RetrievalQA: Configured RAG chain
        """
        # Custom prompt template for better responses
        prompt_template = """You are a helpful AI assistant that answers questions based on the provided context. 
        Use only the information from the context to answer the question. If the context doesn't contain enough 
        information to answer the question, say "I don't have enough information to answer this question based on the provided context."

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return chain
        
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict[str, Any]: Response containing answer and source documents
        """
        if not self.chain:
            raise ValueError("RAG chain not initialized. Please set a retriever first.")
            
        try:
            logger.info(f"Processing question: {question}")
            
            # Get response from chain
            response = self.chain({"query": question})
            
            # Extract answer and source documents
            answer = response.get("result", "No answer generated")
            source_documents = response.get("source_documents", [])
            
            # Format response
            result = {
                "question": question,
                "answer": answer,
                "source_documents": source_documents,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name
            }
            
            logger.info(f"Generated answer with {len(source_documents)} source documents")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "answer": f"Error processing your question: {str(e)}",
                "source_documents": [],
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "error": True
            }
            
    def set_retriever(self, retriever) -> None:
        """
        Set the document retriever for the RAG chain.
        
        Args:
            retriever: Document retriever (e.g., FAISS-based)
        """
        self.retriever = retriever
        self.chain = self._create_chain()
        logger.info("Retriever updated and chain recreated")
        
    def get_chain_info(self) -> Dict[str, Any]:
        """
        Get information about the current RAG chain configuration.
        
        Returns:
            Dict[str, Any]: Chain configuration information
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "has_retriever": self.retriever is not None,
            "chain_type": "RetrievalQA"
        }


class RAGManager:
    """
    A high-level manager for RAG operations.
    
    This class provides a simplified interface for RAG functionality and manages
    the interaction between document retrieval and response generation.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
        """
        Initialize the RAG Manager.
        
        Args:
            model_name (str): OpenAI model to use
            temperature (float): Temperature for generation
            max_tokens (int): Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rag_chain = None
        
    def setup_chain(self, retriever) -> None:
        """
        Set up the RAG chain with a retriever.
        
        Args:
            retriever: Document retriever
        """
        self.rag_chain = RAGChain(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            retriever=retriever
        )
        logger.info("RAG chain setup complete")
        
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a response.
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict[str, Any]: Response with answer and sources
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Please call setup_chain() first.")
            
        return self.rag_chain.query(question)
        
    def format_response(self, response: Dict[str, Any], include_sources: bool = True) -> str:
        """
        Format the RAG response for display.
        
        Args:
            response (Dict[str, Any]): RAG response
            include_sources (bool): Whether to include source information
            
        Returns:
            str: Formatted response string
        """
        if response.get("error"):
            return f"❌ Error: {response['answer']}"
            
        formatted = f"🤖 {response['answer']}\n\n"
        
        if include_sources and response.get("source_documents"):
            formatted += "📚 Sources:\n"
            for i, doc in enumerate(response["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                filename = doc.metadata.get("filename", "Unknown")
                formatted += f"{i}. {filename} (Score: {doc.metadata.get('score', 'N/A')})\n"
                
        return formatted
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get RAG system statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        if not self.rag_chain:
            return {"status": "Not initialized"}
            
        return {
            "status": "Ready",
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chain_info": self.rag_chain.get_chain_info()
        }


class ConversationLogger:
    """
    A class to log and manage conversation history.
    """
    
    def __init__(self, log_file: str = "outputs/conversation_log.jsonl"):
        """
        Initialize the conversation logger.
        
        Args:
            log_file (str): Path to log file
        """
        self.log_file = log_file
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log_interaction(self, response: Dict[str, Any]) -> None:
        """
        Log a Q&A interaction to file.
        
        Args:
            response (Dict[str, Any]): RAG response to log
        """
        import json
        
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
            
    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent interactions from log.
        
        Args:
            limit (int): Number of recent interactions to retrieve
            
        Returns:
            List[Dict[str, Any]]: Recent interactions
        """
        import json
        
        if not self.log_path.exists():
            return []
            
        interactions = []
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    interactions.append(json.loads(line.strip()))
        except Exception as e:
            logger.error(f"Error reading log: {str(e)}")
            
        return interactions


if __name__ == "__main__":
    # Test the RAG chain
    from loaders import DocumentLoader, create_sample_documents
    from embedder import FAISSEmbedder
    
    # Create sample documents
    create_sample_documents()
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    # Create embedder and retriever
    embedder = FAISSEmbedder()
    embedder.create_index(documents)
    
    # Create a simple retriever function
    def simple_retriever(query: str, k: int = 4):
        return [doc for doc, _ in embedder.similarity_search(query, k)]
    
    # Create RAG chain
    rag_chain = RAGChain(retriever=simple_retriever)
    
    # Test query
    response = rag_chain.query("What is machine learning?")
    print("Question:", response["question"])
    print("Answer:", response["answer"])
    print("Sources:", len(response["source_documents"])) 