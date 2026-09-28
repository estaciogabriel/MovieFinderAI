"""
Módulo de Clusterização Não Supervisionada
Implementa K-Means, DBSCAN e análise de clusters
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class DocumentClusterer:
    """Classe para clusterização de documentos usando embeddings"""
    
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings = None
        self.documents = None
        self.labels = None
        self.clusterer = None
        
        self._load_data()
    
    def _load_data(self):
        """Carrega embeddings e documentos"""
        embeddings_path = self.embeddings_dir / 'embeddings.npy'
        documents_path = self.embeddings_dir / 'documents.pkl'
        
        self.embeddings = np.load(embeddings_path)
        
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"✓ Loaded {len(self.documents)} documents")
        print(f"  Embedding shape: {self.embeddings.shape}")
    
    def find_optimal_k(self, k_range: range = range(5, 21), 
                       method: str = 'elbow') -> int:
        """
        Encontra número ótimo de clusters
        
        Args:
            k_range: Range de valores K para testar
            method: 'elbow' ou 'silhouette'
        
        Returns:
            Número ótimo de clusters
        """
        print(f"\nFinding optimal K using {method} method...")
        
        inertias = []
        silhouettes = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                sil_score = silhouette_score(self.embeddings, labels)
                silhouettes.append(sil_score)
            else:
                silhouettes.append(0)
            
            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.3f}")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(list(k_range), inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (K)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True)
        
        axes[1].plot(list(k_range), silhouettes, 'ro-')
        axes[1].set_xlabel('Number of Clusters (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        output_path = self.embeddings_dir.parent / 'optimal_k_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved analysis plot to {output_path}")
        plt.close()
        
        if method == 'silhouette':
            optimal_k = list(k_range)[np.argmax(silhouettes)]
        else:
            optimal_k = list(k_range)[len(k_range)//3]
        
        print(f"\n✓ Optimal K: {optimal_k}")
        return optimal_k
    
    def cluster_kmeans(self, n_clusters: int = 10, random_state: int = 42) -> np.ndarray:
        """
        Aplica K-Means clustering
        
        Args:
            n_clusters: Número de clusters
            random_state: Seed para reprodutibilidade
        
        Returns:
            Array de labels dos clusters
        """
        print(f"\nApplying K-Means clustering with K={n_clusters}...")
        
        self.clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        self.labels = self.clusterer.fit_predict(self.embeddings)
        
        silhouette = silhouette_score(self.embeddings, self.labels)
        calinski = calinski_harabasz_score(self.embeddings, self.labels)
        davies = davies_bouldin_score(self.embeddings, self.labels)
        
        print(f"\n✓ K-Means clustering complete")
        print(f"  Silhouette Score: {silhouette:.3f} (higher is better)")
        print(f"  Calinski-Harabasz: {calinski:.2f} (higher is better)")
        print(f"  Davies-Bouldin: {davies:.3f} (lower is better)")
        
        cluster_counts = Counter(self.labels)
        print(f"\nCluster distribution:")
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            percentage = (count / len(self.labels)) * 100
            print(f"  Cluster {cluster_id}: {count} docs ({percentage:.1f}%)")
        
        return self.labels
    
    def cluster_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Aplica DBSCAN clustering (densidade)
        
        Args:
            eps: Distância máxima entre pontos
            min_samples: Mínimo de amostras por cluster
        
        Returns:
            Array de labels (-1 = outlier)
        """
        print(f"\nApplying DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        self.labels = self.clusterer.fit_predict(self.embeddings)
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        print(f"\n✓ DBSCAN clustering complete")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points (outliers): {n_noise} ({n_noise/len(self.labels)*100:.1f}%)")
        
        if n_clusters > 1:
            mask = self.labels != -1
            if mask.sum() > 1:
                silhouette = silhouette_score(self.embeddings[mask], self.labels[mask])
                print(f"  Silhouette Score: {silhouette:.3f}")
        
        return self.labels
    
    def cluster_hierarchical(self, n_clusters: int = 10, linkage: str = 'ward') -> np.ndarray:
        """
        Aplica Clustering Hierárquico
        
        Args:
            n_clusters: Número de clusters
            linkage: Tipo de linkage ('ward', 'complete', 'average')
        
        Returns:
            Array de labels
        """
        print(f"\nApplying Hierarchical clustering (n_clusters={n_clusters}, linkage={linkage})...")
        
        self.clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        self.labels = self.clusterer.fit_predict(self.embeddings)
        
        silhouette = silhouette_score(self.embeddings, self.labels)
        
        print(f"\n✓ Hierarchical clustering complete")
        print(f"  Silhouette Score: {silhouette:.3f}")
        
        return self.labels
    
    def analyze_clusters(self, top_n_terms: int = 10) -> pd.DataFrame:
        """
        Analisa conteúdo dos clusters
        
        Args:
            top_n_terms: Número de termos mais comuns por cluster
        
        Returns:
            DataFrame com análise dos clusters
        """
        if self.labels is None:
            raise ValueError("Run clustering first!")
        
        print(f"\nAnalyzing cluster content...")
        
        cluster_analysis = []
        
        unique_labels = sorted(set(self.labels))
        
        for cluster_id in unique_labels:
            if cluster_id == -1:
                cluster_name = "Outliers"
            else:
                cluster_name = f"Cluster {cluster_id}"
            
            mask = self.labels == cluster_id
            cluster_docs = [self.documents[i] for i in np.where(mask)[0]]
            
            n_docs = len(cluster_docs)
            
            sample_titles = []
            for doc in cluster_docs[:5]:
                text = doc['text']
                first_line = text.split('\n')[0] if '\n' in text else text[:100]
                sample_titles.append(first_line)
            
            cluster_analysis.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'n_documents': n_docs,
                'percentage': (n_docs / len(self.labels)) * 100,
                'sample_titles': sample_titles
            })
        
        df = pd.DataFrame(cluster_analysis)
        
        print(f"\n✓ Cluster analysis complete")
        print(f"\n{df[['cluster_name', 'n_documents', 'percentage']]}")
        
        return df
    
    def get_cluster_representatives(self, n_representatives: int = 5) -> Dict[int, List[Dict]]:
        """
        Encontra documentos mais representativos de cada cluster (mais próximos do centroide)
        
        Args:
            n_representatives: Número de representantes por cluster
        
        Returns:
            Dicionário {cluster_id: [documentos representativos]}
        """
        if self.labels is None or self.clusterer is None:
            raise ValueError("Run K-Means clustering first!")
        
        if not hasattr(self.clusterer, 'cluster_centers_'):
            raise ValueError("Only K-Means supports centroids!")
        
        print(f"\nFinding {n_representatives} representative documents per cluster...")
        
        representatives = {}
        
        for cluster_id in range(self.clusterer.n_clusters):
            mask = self.labels == cluster_id
            cluster_indices = np.where(mask)[0]
            cluster_embeddings = self.embeddings[mask]
            
            centroid = self.clusterer.cluster_centers_[cluster_id]
            
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            
            closest_indices = np.argsort(distances)[:n_representatives]
            
            reps = []
            for idx in closest_indices:
                doc_idx = cluster_indices[idx]
                doc = self.documents[doc_idx]
                reps.append({
                    'filename': doc['filename'],
                    'text': doc['text'],
                    'distance_to_centroid': distances[idx]
                })
            
            representatives[cluster_id] = reps
        
        print(f"✓ Found representatives for {len(representatives)} clusters")
        
        return representatives
    
    def save_clusters(self, output_path: str):
        """Salva resultados da clusterização"""
        if self.labels is None:
            raise ValueError("Run clustering first!")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cluster_data = []
        for i, (doc, label) in enumerate(zip(self.documents, self.labels)):
            cluster_data.append({
                'filename': doc['filename'],
                'cluster_id': int(label),
                'text_preview': doc['text'][:200]
            })
        
        df = pd.DataFrame(cluster_data)
        
        csv_path = output_path.parent / f"{output_path.stem}.csv"
        df.to_csv(csv_path, index=False)
        
        np.save(output_path, self.labels)
        
        print(f"\n✓ Saved clustering results:")
        print(f"  - {csv_path}")
        print(f"  - {output_path}")

def main():
    """Exemplo de uso"""
    base_dir = Path(__file__).parent.parent
    embeddings_dir = base_dir / 'data/processed/embeddings'
    
    clusterer = DocumentClusterer(str(embeddings_dir))
    
    optimal_k = clusterer.find_optimal_k(k_range=range(5, 16))
    
    labels = clusterer.cluster_kmeans(n_clusters=optimal_k)
    
    analysis = clusterer.analyze_clusters()
    
    representatives = clusterer.get_cluster_representatives(n_representatives=3)
    
    print("\n" + "=" * 80)
    print("CLUSTER REPRESENTATIVES")
    print("=" * 80)
    
    for cluster_id, reps in representatives.items():
        print(f"\n📌 Cluster {cluster_id}:")
        for i, rep in enumerate(reps, 1):
            title = rep['text'].split('\n')[0]
            print(f"  {i}. {title}")
            print(f"     Distance: {rep['distance_to_centroid']:.4f}")
    
    output_path = base_dir / 'data/processed/cluster_labels.npy'
    clusterer.save_clusters(output_path)
    
    print("\n✓ Clustering pipeline complete!")

if __name__ == "__main__":
    main()
