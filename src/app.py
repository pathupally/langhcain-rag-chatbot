"""
Main Application Module

This module provides the main interface for the RAG chatbot, including both CLI and Streamlit interfaces.
It orchestrates all components (document loading, embedding, RAG chain) to create a complete
question-answering system.

Usage:
    CLI: python src/app.py
    Streamlit: streamlit run src/app.py
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from loaders import DocumentLoader, create_sample_documents
from embedder import FAISSEmbedder
from rag_chain import RAGManager, ConversationLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    Main RAG Chatbot class that orchestrates all components.
    
    This class provides a unified interface for document loading, indexing,
    and question answering using the RAG pipeline.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 model_name: str = "gpt-3.5-turbo",
                 embedding_model: str = "text-embedding-ada-002",
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
        """
        Initialize the RAG Chatbot.
        
        Args:
            data_dir (str): Directory containing source documents
            model_name (str): OpenAI model for generation
            embedding_model (str): OpenAI model for embeddings
            temperature (float): Temperature for generation
            max_tokens (int): Maximum tokens in response
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize components
        self.document_loader = DocumentLoader(data_dir=data_dir)
        self.embedder = FAISSEmbedder(model_name=embedding_model)
        self.rag_manager = RAGManager(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.conversation_logger = ConversationLogger()
        
        # System state
        self.is_initialized = False
        
    def initialize(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the RAG system by loading documents and setting up the index.
        
        Args:
            force_rebuild (bool): Force rebuild the index even if it exists
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing RAG Chatbot...")
            
            # Load documents
            documents = self.document_loader.load_documents()
            if not documents:
                logger.warning("No documents found. Creating sample documents...")
                create_sample_documents()
                documents = self.document_loader.load_documents()
                
            if not documents:
                logger.error("No documents available for indexing")
                return False
                
            # Set up embedding index
            if not force_rebuild and self.embedder.load_index():
                logger.info("Loaded existing FAISS index")
            else:
                logger.info("Creating new FAISS index")
                self.embedder.create_index(documents)
                self.embedder.save_index()
                
            # Create retriever function
            def retriever(query: str, k: int = 4):
                results = self.embedder.similarity_search(query, k)
                return [doc for doc, _ in results]
                
            # Set up RAG chain
            self.rag_manager.setup_chain(retriever)
            
            self.is_initialized = True
            logger.info("RAG Chatbot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG Chatbot: {str(e)}")
            return False
            
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a response.
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict[str, Any]: Response with answer and sources
        """
        if not self.is_initialized:
            raise ValueError("RAG Chatbot not initialized. Call initialize() first.")
            
        response = self.rag_manager.ask_question(question)
        
        # Log the interaction
        self.conversation_logger.log_interaction(response)
        
        return response
        
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        stats = {
            "initialized": self.is_initialized,
            "data_directory": self.data_dir,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "document_stats": self.document_loader.get_document_stats(),
            "index_stats": self.embedder.get_index_stats(),
            "rag_stats": self.rag_manager.get_stats()
        }
        
        return stats
        
    def rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index from scratch.
        
        Returns:
            bool: True if rebuild was successful
        """
        try:
            logger.info("Rebuilding FAISS index...")
            documents = self.document_loader.load_documents()
            self.embedder.rebuild_index(documents)
            
            # Recreate retriever
            def retriever(query: str, k: int = 4):
                results = self.embedder.similarity_search(query, k)
                return [doc for doc, _ in results]
                
            self.rag_manager.setup_chain(retriever)
            logger.info("Index rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            return False


def cli_interface():
    """Command-line interface for the RAG chatbot."""
    parser = argparse.ArgumentParser(description="RAG Chatbot - Ask questions about your documents")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens in response")
    parser.add_argument("--data-dir", default="data", help="Directory containing documents")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = RAGChatbot(
        data_dir=args.data_dir,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    if args.rebuild:
        success = chatbot.initialize(force_rebuild=True)
    else:
        success = chatbot.initialize()
        
    if not success:
        print("❌ Failed to initialize RAG Chatbot")
        return
        
    # Print system stats
    stats = chatbot.get_system_stats()
    print(f"\n📊 System Status:")
    print(f"   Documents: {stats['document_stats']['total_files']}")
    print(f"   Index vectors: {stats['index_stats']['total_vectors']}")
    print(f"   Model: {stats['model_name']}")
    print(f"   Status: {'✅ Ready' if stats['initialized'] else '❌ Not Ready'}")
    
    # Interactive Q&A loop
    print(f"\n🤖 RAG Chatbot Ready! Ask questions about your documents.")
    print(f"   Type 'quit', 'exit', or 'q' to exit")
    print(f"   Type 'stats' to see system statistics")
    print(f"   Type 'rebuild' to rebuild the index")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'stats':
                stats = chatbot.get_system_stats()
                print(f"\n📊 System Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            elif question.lower() == 'rebuild':
                print("🔄 Rebuilding index...")
                if chatbot.rebuild_index():
                    print("✅ Index rebuilt successfully")
                else:
                    print("❌ Failed to rebuild index")
                continue
            elif not question:
                continue
                
            # Get response
            response = chatbot.ask_question(question)
            
            # Format and display response
            formatted = chatbot.rag_manager.format_response(response)
            print(f"\n{formatted}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def streamlit_interface():
    """Streamlit web interface for the RAG chatbot."""
    import streamlit as st
    
    # Page configuration
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    # Sidebar
    with st.sidebar:
        st.title("🤖 RAG Chatbot")
        st.markdown("---")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        model_name = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 100)
        
        # Initialize button
        if st.button("🚀 Initialize System"):
            with st.spinner("Initializing RAG system..."):
                chatbot = RAGChatbot(
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                if chatbot.initialize():
                    st.session_state.chatbot = chatbot
                    st.success("✅ System initialized!")
                else:
                    st.error("❌ Failed to initialize system")
                    
        # System stats
        if st.session_state.chatbot:
            st.markdown("---")
            st.subheader("📊 System Stats")
            stats = st.session_state.chatbot.get_system_stats()
            st.json(stats)
            
        # Rebuild index
        if st.button("🔄 Rebuild Index"):
            if st.session_state.chatbot:
                with st.spinner("Rebuilding index..."):
                    if st.session_state.chatbot.rebuild_index():
                        st.success("✅ Index rebuilt!")
                    else:
                        st.error("❌ Failed to rebuild index")
            else:
                st.warning("Please initialize the system first")
                
    # Main content
    st.title("🤖 RAG Chatbot")
    st.markdown("Ask questions about your documents using AI-powered retrieval and generation.")
    
    # Chat interface
    if st.session_state.chatbot:
        # Chat input
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Get response
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.ask_question(user_question)
                
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response["answer"],
                "sources": response.get("source_documents", [])
            })
            
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show sources if available
                    if message.get("sources"):
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(message["sources"], 1):
                                st.markdown(f"**Source {i}:** {doc.metadata.get('filename', 'Unknown')}")
                                st.markdown(f"**Content:** {doc.page_content[:200]}...")
                                
    else:
        st.info("👈 Please initialize the system using the sidebar to start chatting!")
        
        # Show sample questions
        st.markdown("---")
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What is machine learning?",
            "How does RAG work?",
            "What are the benefits of retrieval-augmented generation?",
            "What is the difference between AI and machine learning?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.info(f"Please initialize the system first, then ask: {question}")


def main():
    """Main entry point for the application."""
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=sk-your-api-key-here")
        return
        
    # Check if running with Streamlit
    if "streamlit" in sys.modules:
        streamlit_interface()
    else:
        cli_interface()


if __name__ == "__main__":
    main() 