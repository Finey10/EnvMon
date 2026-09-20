"""
RAG Pipeline using LangChain and ChromaDB

Implements:
1. Document Loader (TextLoader)
2. Text Splitter (RecursiveCharacterTextSplitter)
3. Embeddings (HuggingFaceEmbeddings / sentence-transformers)
4. Vector Store (Chroma)
5. Retriever
"""

from pathlib import Path

# Langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

_RETRIEVER = None

def get_policy_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        file_path = Path(__file__).parent.parent / "data" / "env_policies.txt"
        
        # 1. Loader
        loader = TextLoader(str(file_path))
        docs = loader.load()
        
        # 2. Splitter 
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n"]
        )
        chunks = splitter.split_documents(docs)
        
        # 3. Embeddings (Lightweight local model)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 4. Vector Store
        vector_store = Chroma.from_documents(chunks, embeddings, collection_name="env_policies")
        
        # 5. Retriever
        _RETRIEVER = vector_store.as_retriever(search_kwargs={"k": 1})
        
    return _RETRIEVER


def retrieve_policy(query: str) -> str:
    try:
        retriever = get_policy_retriever()
        docs = retriever.invoke(query)
        if docs:
            return docs[0].page_content
    except Exception as e:
        print(f"[RAG Error] {e}")
    return "No specific environmental policy matched this query."
