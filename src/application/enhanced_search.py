"""
Busca aprimorada com clustering e detecção de anomalias
"""

from src.infrastructure.embedder import DocumentEmbedder
from src.infrastructure.vector_db import VectorDatabase
from src.infrastructure.clustering import DocumentClusterer
from src.infrastructure.anomaly_detection import AnomalyDetector
import numpy as np

class EnhancedSearch:
    def __init__(self, chroma_dir, embeddings_dir):
        self.embedder = DocumentEmbedder(model_name='all-mpnet-base-v2')
        self.db = VectorDatabase(db_path=chroma_dir, collection_name="movies_docs")
        self.db.create_collection(reset=False)
        
        self.clusterer = DocumentClusterer(embeddings_dir)
        self.clusterer.cluster_kmeans(n_clusters=20)
        
        self.detector = AnomalyDetector(embeddings_dir)
        self.detector.detect_isolation_forest(contamination=0.05)
    
    def search_with_recommendations(self, query, n_results=5, n_similar=3):
        """Busca com recomendações do mesmo cluster"""
        query_emb = self.embedder.model.encode([query], normalize_embeddings=True)[0]
        results = self.db.search(query_embedding=query_emb, n_results=n_results)
        
        if not results['documents'][0]:
            return {'main_results': [], 'recommendations': []}
        
        first_doc_idx = self._get_doc_index(results['metadatas'][0][0]['filename'])
        cluster_id = self.clusterer.labels[first_doc_idx]
        
        cluster_docs = np.where(self.clusterer.labels == cluster_id)[0]
        recommendations = []
        
        for idx in cluster_docs[:n_similar + 1]:
            if idx != first_doc_idx:
                doc = self.clusterer.documents[idx]
                recommendations.append({
                    'filename': doc['filename'],
                    'text': doc['text'][:200],
                    'cluster': f"Cluster {cluster_id}"
                })
                if len(recommendations) >= n_similar:
                    break
        
        return {
            'main_results': results,
            'recommendations': recommendations,
            'cluster_info': f"Cluster {cluster_id} - {len(cluster_docs)} filmes similares"
        }
    
    def search_with_quality_filter(self, query, n_results=10):
        """Busca filtrando anomalias (documentos ruins)"""
        query_emb = self.embedder.model.encode([query], normalize_embeddings=True)[0]
        results = self.db.search(query_embedding=query_emb, n_results=n_results * 2)
        
        # Filtrar anomalias
        filtered_results = {
            'documents': [[]],
            'distances': [[]],
            'metadatas': [[]]
        }
        
        for doc, dist, meta in zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ):
            doc_idx = self._get_doc_index(meta['filename'])
            
            if not self.detector.is_anomaly[doc_idx]:
                filtered_results['documents'][0].append(doc)
                filtered_results['distances'][0].append(dist)
                filtered_results['metadatas'][0].append(meta)
                
                if len(filtered_results['documents'][0]) >= n_results:
                    break
        
        return filtered_results
    
    def _get_doc_index(self, filename):
        """Encontra índice do documento"""
        for i, doc in enumerate(self.clusterer.documents):
            if doc['filename'] == filename:
                return i
        return 0

def main():
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    search = EnhancedSearch(
        chroma_dir=os.path.join(base_dir, 'chroma_db'),
        embeddings_dir=os.path.join(base_dir, 'data/processed/embeddings')
    )
    
    print("=== BUSCA COM RECOMENDAÇÕES ===")
    results = search.search_with_recommendations("Titanic ship romance", n_results=3, n_similar=3)
    
    print("\n📌 Resultado Principal:")
    for i, doc in enumerate(results['main_results']['documents'][0][:1], 1):
        print(f"{i}. {doc.split(chr(10))[0]}")
    
    print(f"\n🎬 Recomendações ({results['cluster_info']}):")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec['text'].split(chr(10))[0]}")
    
    print("\n\n=== BUSCA COM FILTRO DE QUALIDADE ===")
    filtered = search.search_with_quality_filter("action movie", n_results=5)
    
    print("\n✅ Resultados (sem anomalias):")
    for i, doc in enumerate(filtered['documents'][0], 1):
        print(f"{i}. {doc.split(chr(10))[0]}")

if __name__ == "__main__":
    main()
