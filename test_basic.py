#!/usr/bin/env python3
"""
Basic test script to verify core functionality without OpenAI API key.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loaders import DocumentLoader, create_sample_documents

def test_document_loading():
    """Test document loading functionality."""
    print("🧪 Testing Document Loading...")
    
    # Create sample documents
    create_sample_documents()
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    print(f"✅ Loaded {len(documents)} document chunks")
    
    # Check document stats
    stats = loader.get_document_stats()
    print(f"✅ Document stats: {stats}")
    
    # Display sample content
    if documents:
        print(f"\n📄 Sample document content:")
        print(f"Content: {documents[0].page_content[:200]}...")
        print(f"Metadata: {documents[0].metadata}")
    
    return True

def test_project_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing Project Structure...")
    
    required_files = [
        "requirements.txt",
        "env.example",
        "README.md",
        "LICENSE",
        "src/loaders.py",
        "src/embedder.py",
        "src/rag_chain.py",
        "src/app.py",
        "docs/API_REFERENCE.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_directory_structure():
    """Test that required directories exist."""
    print("\n🧪 Testing Directory Structure...")
    
    required_dirs = ["data", "docs", "outputs", "src"]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            return False
    
    print("✅ All required directories present")
    return True

def main():
    """Run all tests."""
    print("🚀 Starting Basic Functionality Tests\n")
    
    tests = [
        test_project_structure,
        test_directory_structure,
        test_document_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The RAG chatbot is ready for setup.")
        print("\n📝 Next steps:")
        print("1. Copy env.example to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Run: python src/app.py")
        print("4. Or run: streamlit run src/app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 