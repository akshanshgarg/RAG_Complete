from src.dataloader import DataLoader
from src.embedding import Embedding


if __name__ == "__main__":
    # Load data
    data_loader = DataLoader(data_dir='data')
    documents = data_loader.load_data()

    # Initialize embedding model
    embedder = Embedding()

    

    print(f"Loaded {len(documents)} documents.")

