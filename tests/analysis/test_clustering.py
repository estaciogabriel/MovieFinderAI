"""
Testes unitários para o módulo de clustering
"""

import pytest
import numpy as np
from pathlib import Path


from src.infrastructure.clustering import DocumentClusterer

@pytest.fixture
def clusterer():
    """Fixture para criar clusterer com dados de teste"""
    base_dir = Path(__file__).resolve().parents[2]
    embeddings_dir = base_dir / 'data/processed/embeddings'
    
    if not embeddings_dir.exists():
        pytest.skip("Embeddings não encontrados. Execute o pipeline primeiro.")
    
    return DocumentClusterer(str(embeddings_dir))

def test_clusterer_initialization(clusterer):
    """Testa inicialização do clusterer"""
    assert clusterer.embeddings is not None
    assert clusterer.documents is not None
    assert len(clusterer.embeddings) == len(clusterer.documents)

def test_kmeans_clustering(clusterer):
    """Testa clustering K-Means"""
    labels = clusterer.cluster_kmeans(n_clusters=5)
    
    assert labels is not None
    assert len(labels) == len(clusterer.embeddings)
    assert len(set(labels)) == 5

def test_cluster_analysis(clusterer):
    """Testa análise de clusters"""
    clusterer.cluster_kmeans(n_clusters=5)
    analysis = clusterer.analyze_clusters()
    
    assert len(analysis) == 5
    assert 'cluster_name' in analysis.columns
    assert 'n_documents' in analysis.columns

def test_cluster_representatives(clusterer):
    """Testa busca de representantes"""
    clusterer.cluster_kmeans(n_clusters=3)
    reps = clusterer.get_cluster_representatives(n_representatives=2)
    
    assert len(reps) == 3
    for cluster_id, documents in reps.items():
        assert len(documents) == 2
