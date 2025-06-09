# 🤖 LangChain Local RAG Chatbot

A fully self-contained Retrieval-Augmented Generation (RAG) chatbot built with LangChain, OpenAI, and FAISS. This application allows you to load documents (PDF or TXT files), index them using vector embeddings, and ask questions about their content using natural language.

## 🚀 Features

- **Document Loading**: Support for PDF and TXT files with automatic chunking
- **Vector Search**: FAISS-based similarity search for efficient document retrieval
- **RAG Pipeline**: Combines retrieval with OpenAI's language models for accurate answers
- **Dual Interface**: Both command-line and Streamlit web interfaces
- **Source Tracking**: See which documents were used to generate each answer
- **Conversation Logging**: Automatic logging of all Q&A interactions
- **Modular Design**: Clean, well-documented, and easily extensible codebase
- **Persistent Index**: FAISS index is saved and can be reused across sessions
- **Error Handling**: Comprehensive error handling and graceful degradation

## 📋 Requirements

- **Python**: 3.10+ (tested with Python 3.13)
- **OpenAI API Key**: Required for embeddings and language model
- **Memory**: 2GB+ RAM (for FAISS indexing)
- **Storage**: ~100MB for dependencies + space for your documents

## 🛠 Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd langchain-local-rag-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your OpenAI API key
nano .env  # or use your preferred editor
```

Add your OpenAI API key to the `.env` file:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL_NAME=gpt-4.1
```

### 3. Add Your Documents

Place your documents in the `data/` directory:
```bash
data/
├── document1.pdf
├── document2.txt
├── research_paper.pdf
└── manual.txt
```

## 🎯 Quick Start

### Command Line Interface

```bash
# Basic usage
python src/app.py

# With custom options
python src/app.py --model gpt-4 --temperature 0.2 --rebuild

# Available CLI options
python src/app.py --help
```

### Streamlit Web Interface

```bash
streamlit run src/app.py
```

The web interface will open at `http://localhost:8501`

## 📁 Project Structure

```
langchain-local-rag-chatbot/
├── data/                    # 📁 Put your source PDFs or text files here
│   ├── ai_ml_overview.txt   # Sample document
│   └── rag_system.txt       # Sample document
├── docs/                    # 📁 Project documentation
│   └── API_REFERENCE.md     # Detailed API documentation
├── outputs/                 # 📁 Logs, evaluation, and FAISS index
│   ├── faiss_index         # Vector database (auto-created)
│   ├── documents.pkl       # Document metadata (auto-created)
│   └── conversation_log.jsonl # Q&A history (auto-created)
├── src/                     # 📁 Source code
│   ├── loaders.py          # Document loading and chunking
│   ├── embedder.py         # FAISS embedding and vector search
│   ├── rag_chain.py        # RAG chain construction
│   └── app.py              # Main application (CLI + Streamlit)
├── venv/                    # 📁 Virtual environment (auto-created)
├── .env.example            # 📄 Example environment file
├── requirements.txt        # 📄 Required packages
├── test_basic.py          # 📄 Basic functionality tests
├── README.md              # 📄 This file
└── LICENSE                # 📄 MIT License
```

## 🔧 Detailed Usage

### 1. Document Preparation

The system supports the following file types:
- **PDF files** (`.pdf`) - Using PyPDF for extraction
- **Text files** (`.txt`) - UTF-8 encoded

**Document Processing:**
- Documents are automatically chunked into smaller pieces (default: 1000 characters)
- Chunks overlap by 200 characters to maintain context
- Each chunk is embedded and stored in the FAISS index

### 2. Running the Application

#### CLI Mode Features

```bash
python src/app.py
```

**Interactive Commands:**
- `quit`, `exit`, `q` - Exit the application
- `stats` - View system statistics
- `rebuild` - Rebuild the FAISS index

**Command Line Options:**
```bash
python src/app.py --help
```

Available options:
- `--rebuild` - Force rebuild the FAISS index
- `--model` - Choose OpenAI model (gpt-3.5-turbo, gpt-4, gpt-4-turbo)
- `--temperature` - Control response creativity (0.0-1.0)
- `--max-tokens` - Maximum response length
- `--data-dir` - Custom data directory path

#### Streamlit Mode Features

```bash
streamlit run src/app.py
```

**Web Interface Features:**
- Chat-like interaction with message history
- Real-time system statistics in sidebar
- Source document viewing with expandable sections
- Configuration options for model and parameters
- Index rebuild functionality
- Sample questions for testing

### 3. Sample Questions and Expected Responses

Based on the included sample documents:

**Q: "What is machine learning?"**
**A:** Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

**Q: "How does RAG work?"**
**A:** RAG combines document retrieval with language model generation. The process involves: 1) Loading and chunking documents, 2) Generating embeddings, 3) Storing in vector database, 4) Retrieving relevant chunks for queries, 5) Using retrieved context to generate accurate answers.

**Q: "What are the benefits of RAG?"**
**A:** RAG provides up-to-date information, reduces hallucinations, improves accuracy with domain-specific knowledge, enables traceability to source documents, and allows for easy knowledge updates.

## ⚙️ Configuration Options

### Environment Variables

```env
# Required
OPENAI_API_KEY=sk-your-api-key

# Optional
OPENAI_MODEL_NAME=gpt-3.5-turbo
```

### Advanced Configuration

You can modify the following parameters in the source code:

**Document Chunking** (`src/loaders.py`):
```python
DocumentLoader(
    chunk_size=1000,      # Size of text chunks
    chunk_overlap=200     # Overlap between chunks
)
```

**Embedding Model** (`src/embedder.py`):
```python
FAISSEmbedder(
    model_name="text-embedding-ada-002"  # OpenAI embedding model
)
```

**RAG Chain** (`src/rag_chain.py`):
```python
RAGChain(
    model_name="gpt-3.5-turbo",  # OpenAI language model
    temperature=0.1,             # Response creativity
    max_tokens=1000             # Maximum response length
)
```

## 🔍 How It Works

### 1. Document Processing Pipeline

```
Documents → Chunking → Embedding → FAISS Index → Retrieval → RAG Chain → Response
```

**Step-by-step process:**

1. **Document Loading**: Documents are loaded using LangChain's document loaders
2. **Text Chunking**: Large documents are split into manageable chunks using RecursiveCharacterTextSplitter
3. **Embedding Generation**: Text chunks are converted to vectors using OpenAI's embedding model
4. **Vector Storage**: Embeddings are stored in a FAISS index for fast similarity search
5. **Query Processing**: User questions are embedded and similar document chunks are retrieved
6. **Answer Generation**: Retrieved context is combined with the question to generate accurate answers
7. **Source Attribution**: The system tracks which documents were used for each answer

### 2. Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   FAISS         │    │   OpenAI        │
│   Loader        │───▶│   Embedder      │───▶│   RAG Chain     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text          │    │   Vector        │    │   Response      │
│   Chunks        │    │   Search        │    │   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📈 Performance Tips

### Optimization Strategies

1. **Chunk Size Tuning**:
   - Smaller chunks (500-800 chars): Better for specific questions
   - Larger chunks (1000-1500 chars): Better for comprehensive answers
   - Overlap: 10-20% of chunk size for context continuity

2. **Model Selection**:
   - **GPT-3.5-turbo**: Fast, cost-effective, good for most use cases
   - **GPT-4**: Better reasoning, more accurate for complex questions
   - **GPT-4-turbo**: Latest model, best performance but higher cost

3. **Temperature Settings**:
   - **0.0-0.3**: Factual, consistent answers
   - **0.4-0.7**: Balanced creativity and accuracy
   - **0.8-1.0**: More creative, varied responses

4. **Retrieval Optimization**:
   - Adjust `k` parameter (number of retrieved chunks)
   - Set appropriate score threshold for relevance filtering
   - Use semantic search for better context matching

### Memory and Storage

- **FAISS Index**: ~100MB for 10,000 document chunks
- **Embeddings**: ~1.5KB per chunk (1536 dimensions)
- **Processing Time**: ~1-2 seconds per question (depending on model)

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "OPENAI_API_KEY not found"
```bash
# Solution: Check your .env file
cat .env
# Should contain: OPENAI_API_KEY=sk-your-api-key-here
```

#### 2. "No documents found"
```bash
# Solution: Add documents to data/ directory
ls data/
# Should show your PDF/TXT files
```

#### 3. "Error loading index"
```bash
# Solution: Rebuild the index
python src/app.py --rebuild
# Or delete outputs/ directory and restart
rm -rf outputs/
python src/app.py
```

#### 4. "Out of memory"
```bash
# Solution: Reduce chunk size in src/loaders.py
chunk_size=500  # Instead of 1000
```

#### 5. "Module not found"
```bash
# Solution: Activate virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

### Debug Mode

Enable detailed logging:
```python
# Add to any module
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Check system statistics:
```bash
# In CLI mode, type 'stats'
# Or check the Streamlit sidebar
```

## 🔄 Advanced Usage

### Custom Document Loaders

Add support for new file types:

```python
from langchain_community.document_loaders import YourCustomLoader

def _load_single_file(self, file_path: Path) -> List[Document]:
    if file_path.suffix.lower() == '.your_format':
        loader = YourCustomLoader(str(file_path))
        raw_docs = loader.load()
        # ... rest of processing
```

### Custom Embedding Models

Use different embedding providers:

```python
from langchain_openai import AzureOpenAIEmbeddings

self.embeddings = AzureOpenAIEmbeddings(
    azure_deployment="your-deployment",
    openai_api_version="2023-05-15"
)
```

### Persistent Index Management

The FAISS index is automatically saved and can be reused:

```python
# Index is saved to outputs/faiss_index
# Documents metadata to outputs/documents.pkl
# Conversation logs to outputs/conversation_log.jsonl
```

### Batch Processing

For large document collections:

```python
# Process documents in batches
loader = DocumentLoader(chunk_size=500)
documents = loader.load_documents()

# Rebuild index periodically
embedder.rebuild_index(documents)
```

## 📝 Logging and Outputs

### Generated Files

- **`outputs/faiss_index`**: FAISS vector database
- **`outputs/documents.pkl`**: Document metadata and content
- **`outputs/conversation_log.jsonl`**: All Q&A interactions

### Log Format

```json
{
  "question": "What is machine learning?",
  "answer": "Machine Learning is a subset of AI...",
  "source_documents": [...],
  "timestamp": "2024-01-15T10:30:00",
  "model": "gpt-3.5-turbo"
}
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Run tests: `python test_basic.py`
6. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions and classes
- Include type hints where possible
- Add logging for important operations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [OpenAI](https://openai.com/) for language models and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Streamlit](https://streamlit.io/) for the web interface
- [PyPDF](https://pypdf.readthedocs.io/) for PDF processing

## 📞 Support

### Getting Help

1. **Check the troubleshooting section** above
2. **Review the logs** in the `outputs/` directory
3. **Run the test script**: `python test_basic.py`
4. **Check the API reference**: `docs/API_REFERENCE.md`

### Common Questions

**Q: Can I use this with my own documents?**
A: Yes! Just place your PDF or TXT files in the `data/` directory.

**Q: How much does it cost?**
A: Costs depend on your OpenAI API usage. Embeddings are ~$0.0001 per 1K tokens, and GPT-3.5-turbo is ~$0.002 per 1K tokens.

**Q: Can I use a different embedding model?**
A: Yes, you can modify the `FAISSEmbedder` class to use different embedding providers.

**Q: How do I scale this for large document collections?**
A: Consider using batch processing, reducing chunk sizes, or implementing document filtering.

---

## 🎉 Quick Test

Test the system without setting up API keys:

```bash
python test_basic.py
```

This will verify that all components are working correctly.

---

**Happy Chatting! 🤖✨**

*Built with ❤️ using LangChain, OpenAI, and FAISS* 
