"""
RAG Pipeline (Retrieval-Augmented Generation)

Implements:
1. Document Loader
2. Text Splitter
3. Embeddings (TF-IDF Vectorization)
4. Vector Store
5. Retriever
"""

import math
import re
from pathlib import Path
from collections import Counter

class TextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> str:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()


class RecursiveTextSplitter:
    def __init__(self, separator="\n\n"):
        self.separator = separator

    def split_text(self, text: str) -> list[str]:
        # Split by double newline to separate paragraphs/policies
        chunks = text.split(self.separator)
        return [c.strip() for c in chunks if c.strip()]


class TFIDFEmbedder:
    """A pure-python TF-IDF Embedder to avoid heavy dependencies."""
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.is_fitted = False

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, documents: list[str]):
        N = len(documents)
        df = Counter()
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1
                
        for token, count in df.items():
            # Standard IDF formula
            self.idf[token] = math.log((1 + N) / (1 + count)) + 1
            
        self.vocab = {token: i for i, token in enumerate(self.idf.keys())}
        self.is_fitted = True

    def embed(self, text: str) -> dict[int, float]:
        """Returns a sparse vector represented as a dictionary {index: weight}"""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vector = {}
        
        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                # TF-IDF calculation
                weight = (count / len(tokens)) * self.idf[token]
                vector[idx] = weight
                
        # Normalize vector (L2 norm)
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            for idx in vector:
                vector[idx] /= norm
                
        return vector


class InMemoryVectorStore:
    def __init__(self, embedder: TFIDFEmbedder):
        self.embedder = embedder
        self.documents = []
        self.vectors = []

    def add_documents(self, chunks: list[str]):
        self.embedder.fit(chunks)
        self.documents = chunks
        for chunk in chunks:
            self.vectors.append(self.embedder.embed(chunk))

    def _cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        
        sum1 = sum(vec1[x] ** 2 for x in vec1.keys())
        sum2 = sum(vec2[x] ** 2 for x in vec2.keys())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
        if not denominator:
            return 0.0
        return float(numerator) / denominator

    def as_retriever(self, k: int = 1):
        def retrieve(query: str) -> list[str]:
            query_vec = self.embedder.embed(query)
            scores = []
            for doc, doc_vec in zip(self.documents, self.vectors):
                score = self._cosine_similarity(query_vec, doc_vec)
                scores.append((score, doc))
            
            # Sort by highest score
            scores.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in scores[:k] if score > 0.0]
            
        return retrieve

# Singleton instantiation for easy import
_RETRIEVER = None

def get_policy_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        file_path = Path(__file__).parent.parent / "data" / "env_policies.txt"
        
        # 1. Loader
        loader = TextLoader(str(file_path))
        raw_text = loader.load()
        
        # 2. Splitter
        splitter = RecursiveTextSplitter()
        chunks = splitter.split_text(raw_text)
        
        # 3. Embeddings
        embedder = TFIDFEmbedder()
        
        # 4. Vector Store
        vector_store = InMemoryVectorStore(embedder)
        vector_store.add_documents(chunks)
        
        # 5. Retriever
        _RETRIEVER = vector_store.as_retriever(k=1)
        
    return _RETRIEVER

def retrieve_policy(query: str) -> str:
    retriever = get_policy_retriever()
    results = retriever(query)
    if results:
        return results[0]
    return "No specific environmental policy matched this query."
