import os
from dotenv import load_dotenv
from typer import prompt
from .vector_store import FaissVectorStore
from .dataloader import DataLoader
from langchain_groq import ChatGroq

class RagSearch:
    def __init__(self, persist_dir:str = '../data/vector_store', embedding_model: str = 'all-MiniLM-L6-v2', llm_model: str = 'openai/gpt-oss-120b'):
        self.vector_store = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
        # Load or build vector store
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vector_store.load()
        else:
            print(f"[WARNING] No existing vector store found at {faiss_path} or {meta_path}. Building the vector store.")
            dataLoader = DataLoader(data_dir='./data/pdfs')
            self.vector_store.build_from_documents(dataLoader.load_data())
            self.vector_store.save()
            
        load_dotenv()
        groq_api_key = os.getenv("GROQ")
        self.llm = ChatGroq(model=llm_model, groq_api_key=groq_api_key,  temperature=0.1)
        print(f"Initialized RAG Search with embedding model: {embedding_model} and LLM model: {llm_model}")
    
    def search_summarize(self, query: str, top_k: int = 5) -> str:
        print("Searching for query")
        # print(f"[INFO] Searching for query: '{query}' with top_k={top_k}")
        results = self.vector_store.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        print(f"[INFO] Retrieved context for LLM: '{context[:200]}...'")
        if not results: 
            print("[INFO] No relevant documents found.")
            return "No relevant documents found."
        
        # # Extract relevant document contents
        # relevant_docs = [res['metadata']['source'] for res in results]
        # print(f"[INFO] Found {len(relevant_docs)} relevant documents: {relevant_docs}")
        
        # Create a prompt for the LLM
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke(prompt)
        return response.content

# Example usage
if __name__ == "__main__":
    load_dotenv()
    rag_search = RagSearch()
    
    user_query = "What is Attention Mechanism?"
    summary = rag_search.search_summarize(user_query)
    print(f"Summary:\n{summary}")