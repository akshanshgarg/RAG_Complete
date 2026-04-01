# from src.dataloader import DataLoader
# from src.vector_store import FaissVectorStore
from src.search import RagSearch
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    rag_search = RagSearch("./data/vector_store")
    user_query = "What is Attention Mechanism?"
    summary = rag_search.search_summarize(user_query)

    print(f"{summary}")
    
 