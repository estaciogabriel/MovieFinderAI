# Movies Knowledge Base - Tests

Este diretório contém testes unitários para o projeto.

## Executar Testes

```bash
# Todos os testes
pytest tests/ -v

# Teste específico
pytest tests/test_clustering.py -v

# Com coverage
pytest --cov=src tests/

# Com relatório HTML
pytest --cov=src --cov-report=html tests/
```

## Estrutura

- `test_clustering.py`: Testes para módulo de clusterização
- `test_anomaly_detection.py`: Testes para detecção de anomalias
- `test_quality_classifier.py`: Testes para classificadores

## Requisitos

Execute primeiro:
```bash
pip install pytest pytest-cov
python pipeline.py  # Gera embeddings necessários
```
