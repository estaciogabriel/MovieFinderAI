"""
Chroma Cloud Repository
Centralized connection to Chroma Cloud database
"""

import chromadb
from src.config import CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE

class ChromaRepository:
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialize()
            cls._instance = instance
        return cls._instance
    
    def _initialize(self):
        missing = [name for name, value in (
            ("CHROMA_API_KEY", CHROMA_API_KEY),
            ("CHROMA_TENANT", CHROMA_TENANT),
            ("CHROMA_DATABASE", CHROMA_DATABASE),
        ) if not value]
        if missing:
            raise RuntimeError(
                f"Missing required configuration: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in your Chroma Cloud "
                "credentials."
            )
        self.client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE
        )
        self.collection = self.client.get_or_create_collection("movies_docs")
    
    def search(self, query_embedding, n_results=5):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
    
    def count(self):
        return self.collection.count()
    
    def get_collection(self):
        return self.collection
