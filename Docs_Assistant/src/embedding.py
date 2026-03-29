from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        if self.model is None:
            raise ValueError("Model not loaded. Please initialize the Embedder with a valid model name.")   
        return self.model.encode(text, show_progress_bar=True)
    