"""
Busca usando Chroma Cloud
"""

from functools import lru_cache

from src.infrastructure.embedder import DocumentEmbedder
from src.infrastructure.chroma_repository import ChromaRepository
from src.application.search_validator import verify_search_query
from src.application.rerank import rerank_results

EMBEDDING_MODEL = 'all-mpnet-base-v2'

@lru_cache(maxsize=1)
def _get_embedder():
    """Load the embedding model once per process: instantiating
    SentenceTransformer reloads a ~400MB model from disk, which must
    not happen on every search request."""
    return DocumentEmbedder(model_name=EMBEDDING_MODEL)

def search_movies_cloud(query, n_results=5):
    """Search movies in Chroma Cloud"""

    message = verify_search_query(query)
    if message != "OK!":
        return {
            'documents': [[message]],
            'distances': [[0.0]],
            'metadatas': [[{'error': True}]]
        }

    query_embedding = _get_embedder().model.encode([query], normalize_embeddings=True)[0]

    repo = ChromaRepository()
    # Fetch a wide candidate pool: Chroma's HNSW index is approximate and
    # misses true neighbors on small pools (e.g. Titanic for descriptive
    # queries). 100 candidates keeps recall high before re-ranking.
    pool_size = max(n_results, 100)
    results = repo.search(query_embedding, pool_size)
    results = rerank_results(results)

    # Keep only the top n_results after re-ranking
    if results.get('documents') and len(results['documents'][0]) > n_results:
        results['documents'] = [results['documents'][0][:n_results]]
        results['distances'] = [results['distances'][0][:n_results]]
        results['metadatas'] = [results['metadatas'][0][:n_results]]

    return results
