#!/bin/bash

# Script to run the FastAPI application

echo "Starting MovieFinderAI API..."
echo "API will be available at http://localhost:8000"
echo "Docs available at http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "WARNING: .env not found. Copy .env.example to .env and set CHROMA_API_KEY."
fi

# Run with uvicorn
.venv/bin/python -m uvicorn src.interfaces.api.main:app --host 0.0.0.0 --port 8000 --reload
