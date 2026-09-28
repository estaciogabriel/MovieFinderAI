"""
Testes unitários para detecção de anomalias
"""

import pytest
import numpy as np
from pathlib import Path


from src.infrastructure.anomaly_detection import AnomalyDetector

@pytest.fixture
def detector():
    """Fixture para criar detector"""
    base_dir = Path(__file__).resolve().parents[2]
    embeddings_dir = base_dir / 'data/processed/embeddings'
    
    if not embeddings_dir.exists():
        pytest.skip("Embeddings não encontrados. Execute o pipeline primeiro.")
    
    return AnomalyDetector(str(embeddings_dir))

def test_detector_initialization(detector):
    """Testa inicialização"""
    assert detector.embeddings is not None
    assert detector.documents is not None

def test_isolation_forest_detection(detector):
    """Testa Isolation Forest"""
    scores, is_anomaly = detector.detect_isolation_forest(contamination=0.1)
    
    assert len(scores) == len(detector.embeddings)
    assert len(is_anomaly) == len(detector.embeddings)
    
    anomaly_percentage = is_anomaly.sum() / len(is_anomaly)
    assert 0.05 < anomaly_percentage < 0.15

def test_top_anomalies(detector):
    """Testa busca de top anomalias"""
    detector.detect_isolation_forest(contamination=0.1)
    top_anomalies = detector.get_top_anomalies(n=5)
    
    assert len(top_anomalies) == 5
    
    scores = [a['anomaly_score'] for a in top_anomalies]
    assert scores == sorted(scores)

def test_anomaly_statistics(detector):
    """Testa estatísticas"""
    detector.detect_isolation_forest(contamination=0.1)
    stats = detector.get_anomaly_statistics()
    
    assert 'total_documents' in stats.columns
    assert 'n_anomalies' in stats.columns
    assert stats['total_documents'].iloc[0] > 0
