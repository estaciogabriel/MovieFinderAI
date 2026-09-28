"""
Busca usando Chroma Cloud
"""

from src.infrastructure.embedder import DocumentEmbedder
from src.infrastructure.chroma_repository import ChromaRepository
from src.application.search_validator import verify_search_query

def search_movies_cloud(query, n_results=5):
    """Search movies in Chroma Cloud"""
    
    message = verify_search_query(query)
    if message != "OK!":
        return {
            'documents': [[message]],
            'distances': [[0.0]],
            'metadatas': [[{'error': True}]]
        }
    
    embedder = DocumentEmbedder(model_name='all-mpnet-base-v2')
    query_embedding = embedder.model.encode([query], normalize_embeddings=True)[0]
    
    repo = ChromaRepository()
    results = repo.search(query_embedding, n_results)
    
    return results
