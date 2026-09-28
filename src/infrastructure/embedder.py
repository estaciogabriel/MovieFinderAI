import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class DocumentEmbedder:
    # carrega modelo de embeddings
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def load_documents(self, documents_dir):
        docs_path = Path(documents_dir)
        documents = []

        txt_files = list(docs_path.glob("*.txt"))

        for filepath in txt_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            documents.append({
                'filename': filepath.name,
                'text': text,
                'filepath': str(filepath)
            })

        return documents
    
    def create_embeddings(self, documents):
        texts = [doc['text'] for doc in documents]
        embeddings = self.model.encode(texts, batch_size=32, normalize_embeddings=True)
        return embeddings
    
    def save_embeddings(self, embeddings, documents, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.save(output_path / 'embeddings.npy', embeddings)

        with open(output_path / 'documents.pkl', 'wb') as f:
            pickle.dump(documents, f)

        filenames = [doc['filename'] for doc in documents]
        with open(output_path / 'filenames.txt', 'w') as f:
            f.write('\n'.join(filenames))
    
    def load_embeddings(self, embeddings_dir):
        embeddings_path = Path(embeddings_dir)
        embeddings = np.load(embeddings_path / 'embeddings.npy')

        with open(embeddings_path / 'documents.pkl', 'rb') as f:
            documents = pickle.load(f)

        return embeddings, documents
