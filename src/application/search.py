import os

from src.infrastructure.vector_db import VectorDatabase
from src.infrastructure.embedder import DocumentEmbedder
from src.application.search_validator import verify_search_query

def search_movies(query, n_results=5):
    message = verify_search_query(query)
    if message != "OK!":
        return {
            'documents': [[message]],
            'distances': [[0.0]],
            'metadatas': [[{'error': True}]]
        }

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chroma_dir = os.path.join(base_dir, 'chroma_db')

    embedder = DocumentEmbedder(model_name='all-MiniLM-L6-v2')
    query_embedding = embedder.model.encode([query], normalize_embeddings=True)[0]

    db = VectorDatabase(db_path=chroma_dir, collection_name="movies_docs")
    db.create_collection(reset=False)
    results = db.search(query_embedding=query_embedding, n_results=n_results)
    return results

def main():
    print("\nBusca de Filmes")
    while True:
        query = input("Digite sua busca (ou 'sair'): ").strip()

        if query.lower() in ['sair', 'exit']:
            break

        if not query:
            continue
        
        results = search_movies(query, n_results=5)

        if results['metadatas'][0] and results['metadatas'][0][0].get('error'):
            continue

        if results['documents'] and len(results['documents'][0]) > 0:
            for i, (doc, dist, meta) in enumerate(zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            ), 1):
                lines = doc.split('\n')
                print(f"\n{i}. {lines[0] if lines else doc[:100]}")
                print(f"   Distance: {dist:.2f}")
        else:
            print("Nenhum resultado encontrado")

if __name__ == "__main__":
    main()
