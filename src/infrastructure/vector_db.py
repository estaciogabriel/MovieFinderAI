import chromadb
import numpy as np
from pathlib import Path

class VectorDatabase:

    def __init__(self, db_path=None, collection_name="movies_docs"):
        if db_path is None:
            db_path = "./chroma_db"

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection_name = collection_name
        self.collection = None
    
    def create_collection(self, reset=False):
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Movies documents"}
        )
        return self.collection
    
    def add_documents(self, documents, embeddings, batch_size=100):
        if self.collection is None:
            self.create_collection()

        n_docs = len(documents)

        for i in range(0, n_docs, batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]

            ids = [doc['filename'] for doc in batch_docs]
            texts = [doc['text'] for doc in batch_docs]
            metadatas = [{'filename': doc['filename'], 'filepath': doc['filepath']}
                        for doc in batch_docs]

            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
    
    def search(self, query_text=None, query_embedding=None, n_results=5):
        if self.collection is None:
            raise ValueError("Collection not initialized")

        if query_embedding is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
        elif query_text is not None:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
        else:
            raise ValueError("Need query_text or query_embedding")

        return results


    def get_all_embeddings(self):
        if self.collection is None:
            raise ValueError("Collection not initialized")

        results = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        embeddings = np.array(results['embeddings'])
        documents = results['documents']
        metadatas = results['metadatas']
        ids = results['ids']

        return embeddings, documents, metadatas, ids

    def get_stats(self):
        if self.collection is None:
            return {"error": "Collection not initialized"}

        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_documents': count,
            'db_path': str(self.db_path)
        }
    
    def persist(self):
        """Garante que todos os dados sejam gravados no disco"""
        if hasattr(self.client, '_persist'):
            self.client._persist()
    
    def close(self):
        """Fecha a conexão e persiste dados"""
        self.persist()
        del self.client
        del self.collection
