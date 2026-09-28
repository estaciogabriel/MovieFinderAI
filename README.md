# MovieFinderAI

Semantic search API for movies using embeddings and Chroma Cloud.

## Introduction

A system that allows searching for movies using natural language. Type "action movie with a chase scene" and find similar movies by meaning, not just exact words.

## How It Works

1. 43,970 movies from Kaggle transformed into embeddings (768-dimensional vectors)
2. Stored in Chroma Cloud (vector database)
3. Semantic similarity search using sentence-transformers

## Technologies

- Python 3.12+
- FastAPI (API Framework)
- Sentence-Transformers (all-mpnet-base-v2)
- Chroma Cloud
- Pydantic (Data validation)
- Pytest (Testing)

## Project Structure

```
MovieFinderAI/
├── src/
│   ├── config.py              # Environment-based configuration (.env)
│   ├── schemas/
│   │   └── movie.py           # Pydantic request/response models
│   ├── application/           # Business logic (use cases)
│   │   ├── search.py
│   │   ├── search_cloud.py
│   │   ├── search_validator.py
│   │   └── enhanced_search.py
│   ├── infrastructure/        # External tech: Chroma, embeddings, ML tooling
│   │   ├── chroma_repository.py
│   │   ├── embedder.py
│   │   ├── vector_db.py
│   │   ├── document_generator.py
│   │   ├── clustering.py
│   │   ├── anomaly_detection.py
│   │   ├── evaluate.py
│   │   └── visualizer.py
│   └── interfaces/
│       ├── api/               # REST interface (FastAPI)
│       │   ├── main.py
│       │   └── routes/
│       │       └── movies.py
│       └── pages/             # UI interfaces
│           ├── gradio_app.py
│           └── dashboard.py   # Streamlit
├── scripts/
│   └── pipeline.py            # Download embeddings from Chroma Cloud
├── tests/
│   ├── conftest.py
│   ├── api/                   # API integration tests
│   ├── unit/                  # Unit tests
│   └── analysis/               # Clustering/anomaly tests (need local embeddings)
├── .env.example
├── pyproject.toml
├── Makefile
├── run_api.sh
└── README.md
```

Dependency rule: `interfaces -> application -> infrastructure` (schemas shared).

## Quick Start

### 1. Install Dependencies

```bash
# Create a virtualenv (Python 3.12+)
uv venv

# Install dependencies
make install

# Or directly
uv pip install --python .venv/bin/python -r pyproject.toml
```

### 2. Configure Credentials

```bash
# Copy the template and fill in your Chroma Cloud API key
cp .env.example .env
```

The API key is never committed: `.env` is gitignored and `src/config.py`
reads it via environment variables (loaded with python-dotenv).

### 3. Populate the Database

The vector database starts empty. To populate it with the TMDB dataset
(~44,500 movies):

```bash
# Place the TMDB CSVs in CSVs/ (movies_metadata.csv, credits.csv, keywords.csv)
# CSVs/ is gitignored: do not commit the raw data

make upload
```

- Uploads most popular movies first, so the search becomes useful within
  minutes while the rest continues
- Idempotent (`upsert`): safe to stop and re-run to resume
- Expect roughly 2-3 docs/s on CPU (a few hours for the full dataset)

Check progress anytime:

```bash
curl http://localhost:8000/api/v1/health
# documents_count shows how many movies are indexed
```

### 4. Run the API

```bash
# Development mode (with hot reload)
make run-api

# Or directly with uvicorn
.venv/bin/python -m uvicorn src.interfaces.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
make run-api-prod
```

### 5. Access the API

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

Returns the health status of the API and the REAL state of the vector
database (never hardcoded).

**Response (200 OK, database populated):**
```json
{
  "status": "healthy",
  "version": "1.1.0",
  "database_status": "connected",
  "documents_count": 44506,
  "message": null
}
```

**Other states:**
- `200 OK` + `"database_status": "empty"`: reachable but has no documents
  yet; `message` explains how to populate it
- `503 Service Unavailable` + `"database_status": "unreachable"`: cannot
  reach Chroma Cloud; `message` names the environment variables to check

#### Search Movies (POST)

```http
POST /api/v1/movies/search
Content-Type: application/json

{
  "query": "A movie about a ship that collides with an iceberg",
  "n_results": 5
}
```

Search for movies using semantic search.

**Request Body:**
- `query` (string, required): Search query for movies (max: 500 chars)
- `n_results` (integer, optional): Number of results to return (default: 5, min: 1, max: 20)

**Response (200 OK):**
```json
{
  "query": "A movie about a ship that collides with an iceberg",
  "results": [
    {
      "document": "Battleship (2012)\nGenres: Action, Sci-Fi, Thriller\n...",
      "distance": 0.885,
      "metadata": { "movie_id": 85131, "title": "Battleship", "year": "2012", "genres": "..." }
    },
    { "...": "..." }
  ],
  "count": 5,
  "error": null
}
```

A semantic query describes the movie you want ("a ship that collides
with an iceberg") and the API returns the closest matches by meaning —
Titanic (1997) is among the expected results.

**Error Responses:**
- `422 Unprocessable Entity`: Validation error (query too long, invalid n_results)
- `200 OK with error field`: Business validation error (empty/short query,
  special chars only) or an empty database (the error field explains how
  to populate it)
- `503 Service Unavailable`: Chroma Cloud unreachable (structured detail
  with guidance)
- `500 Internal Server Error`: unexpected failure during search

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

# Run analysis tests only (need embeddings from scripts/pipeline.py)
make test-analysis

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

The application uses Chroma Cloud for vector database storage. Credentials
come from environment variables, loaded from a `.env` file at the project
root (see `.env.example`):

```bash
CHROMA_API_KEY=your-api-key       # required
CHROMA_TENANT=your-tenant-id      # optional (has default)
CHROMA_DATABASE=your-database     # optional (has default)
```

## Other Interfaces

```bash
# Gradio chat app
make run            # src/interfaces/pages/gradio_app.py

# Streamlit dashboard
make run-dashboard  # src/interfaces/pages/dashboard.py
```

## License

MIT License
