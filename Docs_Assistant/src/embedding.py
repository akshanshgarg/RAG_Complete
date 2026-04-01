from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from .dataloader import DataLoader


class EmbeddingPipeline:
    def __init__(self, model_name: str ='all-MiniLM-L6-v2', chunk_size: int = 500, chunk_overlap: int = 50):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Initialized embedding model: {model_name}")
    

    def chunk_document(self, document: List[Any]) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separators=["\n\n", "\n", " ", ""], length_function=len)
        chunks = text_splitter.split_documents(document)
        print(f"[INFO] Split {len(document)} documents into {len(chunks)} chunks.")
        return chunks
    

    def embed_chunks(self, chunk: List[Any]) -> np.ndarray:
        try:
            texts = [chunk.page_content for chunk in chunk]
            print(f"[INFO] Embedding {len(texts)} chunks.")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"[INFO] Generated embeddings for {len(embeddings)} chunks.")
            return embeddings
        
        except Exception as e:
            print(f"Error embedding chunk: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Load data
    data_loader = DataLoader(data_dir='../data/pdfs')
    documents = data_loader.load_data()

    print(f"[INFO] Loaded {len(documents)} documents.")
    # Initialize embedding pipeline
    embedding_pipeline = EmbeddingPipeline()

    # Chunk documents
    chunks = embedding_pipeline.chunk_document(documents)

    # Embed chunks
    embeddings = embedding_pipeline.embed_chunks(chunks)

    print(f"[INFO] Example embedding: {embeddings[0] if len(embeddings) > 0 else None}")
    