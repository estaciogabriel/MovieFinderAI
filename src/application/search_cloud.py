"""
Busca usando Chroma Cloud
"""

from src.infrastructure.embedder import DocumentEmbedder
from src.infrastructure.chroma_repository import ChromaRepository
from src.application.search_validator import verify_search_query
from src.application.rerank import rerank_results

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
    # Fetch a wider candidate pool, re-rank by relevance, return the best
    pool_size = min(n_results * 4, 100)
    results = repo.search(query_embedding, pool_size)
    results = rerank_results(results)

    # Keep only the top n_results after re-ranking
    if results.get('documents') and len(results['documents'][0]) > n_results:
        results['documents'] = [results['documents'][0][:n_results]]
        results['distances'] = [results['distances'][0][:n_results]]
        results['metadatas'] = [results['metadatas'][0][:n_results]]

    return results
