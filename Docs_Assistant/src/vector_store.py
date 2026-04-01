from .dataloader import DataLoader
from .embedding import EmbeddingPipeline
import numpy as np
import faiss
import os
import pickle
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class FaissVectorStore:
    def __init__(self, persist_dir:str = "faiss_store", embedding_model: str = 'all-MiniLM-L6-v2', chunk_size: int = 500, chunk_overlap: int = 50):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model)
        print(f"Loaded embedding model: {embedding_model} ")
    
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents.")
        emb_pip = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pip.chunk_document(documents)
        embeddings = emb_pip.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(embeddings, metadatas)
    
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]=None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        
        if metadatas is not None:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {len(embeddings)} embeddings to the vector store.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved vector store to {self.persist_dir}.")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded vector store from {self.persist_dir}.")
        else:
            print(f"[WARNING] No existing vector store found at {self.persist_dir}. Starting fresh.")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata):
                results.append({"index": idx, "metadata": self.metadata[idx], "distance": dist})
        return results
    

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_embedding = self.model.encode([query_text]).astype(np.float32)
        results = self.search(query_embedding, top_k)
        return results


# Example  usage
if __name__ == "__main__":
    # Load data
    data_loader = DataLoader(data_dir='../data/pdfs')
    documents = data_loader.load_data()

    print(f"[INFO] Loaded {len(documents)} documents.")
    # Initialize vector store
    vector_store = FaissVectorStore(persist_dir='../data/vector_store')

    # Build vector store from documents
    vector_store.build_from_documents(documents)

    # Save vector store
    # vector_store.save()

    # Load vector store
    vector_store.load()

    # Query vector store
    query = "What is the main topic of the document?"
    results = vector_store.query(query, top_k=5)
    
    print(f"[INFO] Query results for: '{query}'")
    for res in results:
        print(f"Index: {res['index']}, Distance: {res['distance']:.4f}, Metadata: {res['metadata']}")