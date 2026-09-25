# MovieFinderAI

Semantic search API for movies using embeddings and Chroma Cloud.

## Introduction

A system that allows searching for movies using natural language. Type "action movie with a chase scene" and find similar movies by meaning, not just exact words.

## How It Works

1. 43,970 movies from Kaggle transformed into embeddings (384-dimensional vectors)
2. Stored in Chroma Cloud (vector database)
3. Semantic similarity search using sentence-transformers

## Technologies

- Python 3.12+
- FastAPI (API Framework)
- Sentence-Transformers (all-MiniLM-L6-v2)
- Chroma Cloud
- Pydantic (Data validation)
- Pytest (Testing)

## Project Structure

```
MovieFinderAI/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── routes/
│   │   ├── __init__.py
│   │   └── movies.py        # Movie search endpoints
│   └── schemas/
│       ├── __init__.py
│       └── movie.py         # Pydantic models
├── movies_knowledge_base/
│   ├── config/
│   │   └── chroma_config.py
│   ├── pipeline.py
│   └── src/
│       ├── application/
│       │   ├── search.py
│       │   ├── search_cloud.py
│       │   ├── search_validator.py
│       │   └── enhanced_search.py
│       ├── data/
│       │   ├── document_generator.py
│       │   └── vector_db.py
│       ├── repository/
│       │   └── chroma_repository.py
│       ├── services/
│       │   ├── embedder.py
│       │   ├── anomaly_detection.py
│       │   ├── clustering.py
│       │   ├── visualizer.py
│       │   └── evaluate.py
│       └── tests/
│           ├── test_anomaly_detection.py
│           ├── test_clustering.py
│           └── test_quality_classifier.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── api/
│   │   └── test_movies_api.py
│   └── unit/
│       ├── __init__.py
│       ├── test_search_validator.py
│       ├── test_embedder.py
│       └── test_chroma_repository.py
├── app.py                     # Original Gradio app (kept for reference)
├── app_dashboard.py
├── main.py
├── requirements.txt
├── pyproject.toml
├── Makefile
├── run_api.sh
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using the Makefile
make install
```

### 2. Run the API

```bash
# Development mode (with hot reload)
make run-api

# Or directly with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
make run-api-prod
```

### 3. Access the API

- **Base URL**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/api/v1/health`

## API Documentation

### Endpoints

#### Health Check

```http
GET /api/v1/health
```

Returns the health status of the API and database connection.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database_status": "connected"
}
```

#### Search Movies (POST)

```http
POST /api/v1/movies/search
Content-Type: application/json

{
  "query": "action movie",
  "n_results": 5
}
```

Search for movies using semantic search.

**Request Body:**
- `query` (string, required): Search query for movies (min: 3 chars, max: 500 chars)
- `n_results` (integer, optional): Number of results to return (default: 5, min: 1, max: 20)

**Response (200 OK):**
```json
{
  "query": "action movie",
  "results": [
    {
      "document": "Movie description or title",
      "distance": 0.1234,
      "metadata": { ... }
    }
  ],
  "count": 5,
  "error": null
}
```

**Error Responses:**
- `422 Unprocessable Entity`: Validation error (query too short/long, invalid n_results)
- `200 OK with error field`: Business validation error (empty query, special chars only)
- `500 Internal Server Error`: Database connection or processing error

#### Search Movies (GET)

```http
GET /api/v1/movies/search?query=action+movie&n_results=5
```

Same as POST but for simpler HTTP clients.

## Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run API tests only
make test-api

# Run tests with coverage
make test-cov

# Or directly with pytest
pytest tests/ -v
```

## Code Quality

```bash
# Linting
make lint

# Format code
make format

# Clean cache
make clean
```

## Configuration

The application uses Chroma Cloud for vector database storage. Configuration is in:

```python
# movies_knowledge_base/config/chroma_config.py
CHROMA_API_KEY = "your-api-key"
CHROMA_TENANT = "your-tenant-id"
CHROMA_DATABASE = "your-database-name"
```

## Environment Variables

You can also use environment variables:

```bash
export CHROMA_API_KEY="your-api-key"
export CHROMA_TENANT="your-tenant-id"
export CHROMA_DATABASE="your-database-name"
```

## Original Gradio App

The original Gradio-based application is still available in `app.py`. To run it:

```bash
python app.py
```

## License

MIT License
