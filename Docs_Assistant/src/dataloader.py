import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_data(self):
        all_files = []
        pdf_dir = Path(self.data_dir)
        
        # Find all pdf files in the directory
        for file in pdf_dir.glob('*.pdf'):
            loader = PyPDFLoader(str(file))
            documents = loader.load()
            for doc in documents:
                doc.metadata['source'] = str(file)  
                doc.metadata['filename'] = file.name  
            all_files.extend(documents)
        return all_files

if __name__ == "__main__":
    data_loader = DataLoader('../data/pdfs')
    documents = data_loader.load_data()
    print(documents)  # Print the first document to verify
    print(f"Loaded {len(documents)} documents.")
    