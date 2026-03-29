from embedding import Embedder
from dataloader import DataLoader
import numpy as np
import uuid
import os
import chromadb
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, collection_name: str = "assistant_knowledge_base", persist_directory: str = "../data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.initialize_chromadb()

    def initialize_chromadb(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.Client(chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory
            ))
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

    
    def add_documents(self, documents: List[Dict[str, Any]], Embedder: Embedder):
        try:
            for doc in documents:
                doc_id = str(uuid.uuid4())
                embedding = Embedder.embed(doc.page_content)
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[doc.metadata],
                    documents=[doc.page_content]
                )
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            raise

    
    def search(self, query: str, embedder: Embedder, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = embedder.embed(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents"]
            )
            return results
        except Exception as e:
            print(f"Error searching in ChromaDB: {e}")
            raise