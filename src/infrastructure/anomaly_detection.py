# Detecção de anomalias em embeddings
# Usa Isolation Forest e LOF pra achar documentos estranhos

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyDetector:
    
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings = None
        self.documents = None
        self.anomaly_scores = None
        self.is_anomaly = None
        self.detector = None
        self._load_data()
    
    def _load_data(self):
        """Carrega embeddings e documentos"""
        embeddings_path = self.embeddings_dir / 'embeddings.npy'
        documents_path = self.embeddings_dir / 'documents.pkl'
        
        self.embeddings = np.load(embeddings_path)
        
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"Carregados {len(self.documents)} documentos")
        print(f"Shape: {self.embeddings.shape}")
    
    def detect_isolation_forest(self, 
                               contamination: float = 0.1,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecção de anomalias usando Isolation Forest
        
        Args:
            contamination: Proporção esperada de outliers (0.1 = 10%)
            random_state: Seed para reprodutibilidade
        
        Returns:
            (anomaly_scores, is_anomaly) - scores negativos indicam anomalias
        """
        print(f"\nRodando Isolation Forest (contamination={contamination})...")
        
        self.detector = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        
        predictions = self.detector.fit_predict(self.embeddings)
        
        self.anomaly_scores = self.detector.score_samples(self.embeddings)
        
        self.is_anomaly = predictions == -1
        
        n_anomalies = self.is_anomaly.sum()
        percentage = (n_anomalies / len(self.embeddings)) * 100
        
        print(f"\nResultados:")
        print(f"  Anomalias: {n_anomalies} ({percentage:.1f}%)")
        print(f"  Normais: {(~self.is_anomaly).sum()} ({100-percentage:.1f}%)")
        print(f"\nScores:")
        print(f"  Mais anômalo: {self.anomaly_scores.min():.4f}")
        print(f"  Média: {self.anomaly_scores.mean():.4f}")
        print(f"  Mais normal: {self.anomaly_scores.max():.4f}")
        
        return self.anomaly_scores, self.is_anomaly
    
    def detect_lof(self, 
                   n_neighbors: int = 20,
                   contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecção de anomalias usando Local Outlier Factor
        
        Args:
            n_neighbors: Número de vizinhos para análise local
            contamination: Proporção esperada de outliers
        
        Returns:
            (anomaly_scores, is_anomaly)
        """
        print(f"\nRodando LOF (vizinhos={n_neighbors}, contamination={contamination})...")
        
        self.detector = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False
        )
        
        predictions = self.detector.fit_predict(self.embeddings)
        
        self.anomaly_scores = self.detector.negative_outlier_factor_
        
        self.is_anomaly = predictions == -1
        
        n_anomalies = self.is_anomaly.sum()
        percentage = (n_anomalies / len(self.embeddings)) * 100
        
        print(f"\nResultados:")
        print(f"  Anomalias: {n_anomalies} ({percentage:.1f}%)")
        print(f"  Normais: {(~self.is_anomaly).sum()} ({100-percentage:.1f}%)")
        
        return self.anomaly_scores, self.is_anomaly
    
    def detect_elliptic_envelope(self, contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecção de anomalias usando Elliptic Envelope (assume distribuição gaussiana)
        
        Args:
            contamination: Proporção esperada de outliers
        
        Returns:
            (anomaly_scores, is_anomaly)
        """
        print(f"\nRodando Elliptic Envelope (contamination={contamination})...")
        
        self.detector = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        
        predictions = self.detector.fit_predict(self.embeddings)
        self.anomaly_scores = self.detector.score_samples(self.embeddings)
        self.is_anomaly = predictions == -1
        
        n_anomalies = self.is_anomaly.sum()
        percentage = (n_anomalies / len(self.embeddings)) * 100
        
        print(f"\nResultados: {n_anomalies} anomalias ({percentage:.1f}%)")
        
        return self.anomaly_scores, self.is_anomaly
    
    def get_top_anomalies(self, n: int = 10) -> List[Dict]:
        """
        Retorna os N documentos mais anômalos
        
        Args:
            n: Número de anomalias a retornar
        
        Returns:
            Lista de documentos anômalos com scores
        """
        if self.anomaly_scores is None:
            raise ValueError("Run anomaly detection first!")
        
        anomalous_indices = np.argsort(self.anomaly_scores)[:n]
        
        anomalies = []
        for idx in anomalous_indices:
            doc = self.documents[idx]
            anomalies.append({
                'filename': doc['filename'],
                'text': doc['text'],
                'anomaly_score': self.anomaly_scores[idx],
                'is_anomaly': self.is_anomaly[idx]
            })
        
        return anomalies
    
    def get_anomaly_statistics(self) -> pd.DataFrame:
        """
        Retorna estatísticas sobre anomalias detectadas
        
        Returns:
            DataFrame com estatísticas
        """
        if self.is_anomaly is None:
            raise ValueError("Run anomaly detection first!")
        
        stats = {
            'total_documents': len(self.documents),
            'n_anomalies': self.is_anomaly.sum(),
            'n_normal': (~self.is_anomaly).sum(),
            'anomaly_percentage': (self.is_anomaly.sum() / len(self.documents)) * 100,
            'min_score': self.anomaly_scores.min(),
            'mean_score': self.anomaly_scores.mean(),
            'max_score': self.anomaly_scores.max(),
            'std_score': self.anomaly_scores.std()
        }
        
        return pd.DataFrame([stats])
    
    def plot_anomaly_distribution(self, output_path: Optional[str] = None):
        """
        Visualiza distribuição de scores de anomalia
        
        Args:
            output_path: Caminho para salvar plot (opcional)
        """
        if self.anomaly_scores is None:
            raise ValueError("Run anomaly detection first!")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].hist(self.anomaly_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].axvline(self.anomaly_scores[self.is_anomaly].max(), 
                          color='red', linestyle='--', label='Anomaly threshold')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        data_for_box = pd.DataFrame({
            'Score': self.anomaly_scores,
            'Type': ['Anomaly' if x else 'Normal' for x in self.is_anomaly]
        })
        sns.boxplot(data=data_for_box, x='Type', y='Score', ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('Anomaly Scores: Normal vs Anomalies')
        axes[0, 1].grid(True, alpha=0.3)
        
        sorted_scores = np.sort(self.anomaly_scores)
        axes[1, 0].plot(sorted_scores, linewidth=2, color='steelblue')
        axes[1, 0].axhline(self.anomaly_scores[self.is_anomaly].max(), 
                          color='red', linestyle='--', label='Anomaly threshold')
        axes[1, 0].set_xlabel('Document Index (sorted)')
        axes[1, 0].set_ylabel('Anomaly Score')
        axes[1, 0].set_title('Sorted Anomaly Scores')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        counts = [self.is_anomaly.sum(), (~self.is_anomaly).sum()]
        labels = ['Anomalies', 'Normal']
        colors = ['#FF6B6B', '#4ECDC4']
        axes[1, 1].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 1].set_title('Proportion of Anomalies')
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.embeddings_dir.parent / 'anomaly_distribution.png'
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved anomaly distribution plot to {output_path}")
        plt.close()
    
    def save_anomalies(self, output_path: str):
        """
        Salva resultados da detecção de anomalias
        
        Args:
            output_path: Caminho para salvar CSV
        """
        if self.is_anomaly is None:
            raise ValueError("Run anomaly detection first!")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        anomaly_data = []
        for i, (doc, score, is_anom) in enumerate(zip(self.documents, self.anomaly_scores, self.is_anomaly)):
            anomaly_data.append({
                'filename': doc['filename'],
                'anomaly_score': float(score),
                'is_anomaly': bool(is_anom),
                'text_preview': doc['text'][:200]
            })
        
        df = pd.DataFrame(anomaly_data)
        
        df = df.sort_values('anomaly_score')
        
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Saved anomaly detection results to {output_path}")
        print(f"  Total: {len(df)} documents")
        print(f"  Anomalies: {df['is_anomaly'].sum()}")
        print(f"  Normal: {(~df['is_anomaly']).sum()}")

def main():
    """Exemplo de uso"""
    base_dir = Path(__file__).parent.parent
    embeddings_dir = base_dir / 'data/processed/embeddings'
    
    detector = AnomalyDetector(str(embeddings_dir))
    
    print("\n" + "=" * 80)
    print("ISOLATION FOREST")
    print("=" * 80)
    scores_if, is_anomaly_if = detector.detect_isolation_forest(contamination=0.05)
    
    top_anomalies = detector.get_top_anomalies(n=10)
    
    print("\n" + "=" * 80)
    print("TOP 10 MOST ANOMALOUS DOCUMENTS")
    print("=" * 80)
    
    for i, anom in enumerate(top_anomalies, 1):
        title = anom['text'].split('\n')[0] if '\n' in anom['text'] else anom['text'][:100]
        print(f"\n{i}. {anom['filename']}")
        print(f"   Score: {anom['anomaly_score']:.4f}")
        print(f"   Title: {title}")
    
    stats = detector.get_anomaly_statistics()
    print("\n" + "=" * 80)
    print("ANOMALY STATISTICS")
    print("=" * 80)
    print(stats.T)
    
    detector.plot_anomaly_distribution()
    
    output_path = base_dir / 'data/processed/anomaly_results.csv'
    detector.save_anomalies(output_path)
    
    print("\n" + "=" * 80)
    print("LOCAL OUTLIER FACTOR (LOF)")
    print("=" * 80)
    detector.detect_lof(n_neighbors=20, contamination=0.05)
    
    print("\n✓ Anomaly detection pipeline complete!")

if __name__ == "__main__":
    main()
