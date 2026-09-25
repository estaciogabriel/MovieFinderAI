"""
Movie search routes for FastAPI
"""

import os
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

# Add movies_knowledge_base to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'movies_knowledge_base')))

from api.schemas.movie import (
    MovieSearchRequest,
    MovieSearchResponse,
    MovieResult,
    HealthCheckResponse
)
from src.application.search_cloud import search_movies_cloud
from src.application.search_validator import verify_search_query

router = APIRouter(prefix="/api/v1", tags=["movies"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        database_status="connected"
    )


@router.post("/movies/search", response_model=MovieSearchResponse, responses={400: {"description": "Invalid request"}})
async def search_movies(request: MovieSearchRequest):
    """
    Search for movies using semantic search
    
    This endpoint performs a semantic search on the movie knowledge base
    using embeddings and returns the most similar movies to the query.
    
    - **query**: The search query (must be at least 3 characters)
    - **n_results**: Number of results to return (default: 5, max: 20)
    """
    # Validate query
    validation_result = verify_search_query(request.query)
    if validation_result != "OK!":
        return MovieSearchResponse(
            query=request.query,
            results=[],
            count=0,
            error=validation_result
        )
    
    try:
        # Perform search
        results = search_movies_cloud(request.query, n_results=request.n_results)
        
        # Parse results
        movie_results = []
        
        if results.get('documents') and len(results['documents']) > 0:
            documents = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            
            for i, (doc, dist) in enumerate(zip(documents, distances)):
                metadata = metadatas[i] if i < len(metadatas) else None
                movie_results.append(MovieResult(
                    document=doc,
                    distance=float(dist),
                    metadata=metadata
                ))
        
        return MovieSearchResponse(
            query=request.query,
            results=movie_results,
            count=len(movie_results)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/movies/search", response_model=MovieSearchResponse)
async def search_movies_get(query: str, n_results: int = 5):
    """
    Search for movies using GET request (for simplicity in testing)
    
    This is a GET version of the search endpoint for easier testing
    from browsers or simple HTTP clients.
    """
    # Validate query
    validation_result = verify_search_query(query)
    if validation_result != "OK!":
        return MovieSearchResponse(
            query=query,
            results=[],
            count=0,
            error=validation_result
        )
    
    try:
        # Perform search
        results = search_movies_cloud(query, n_results=n_results)
        
        # Parse results
        movie_results = []
        
        if results.get('documents') and len(results['documents']) > 0:
            documents = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            
            for i, (doc, dist) in enumerate(zip(documents, distances)):
                metadata = metadatas[i] if i < len(metadatas) else None
                movie_results.append(MovieResult(
                    document=doc,
                    distance=float(dist),
                    metadata=metadata
                ))
        
        return MovieSearchResponse(
            query=query,
            results=movie_results,
            count=len(movie_results)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
