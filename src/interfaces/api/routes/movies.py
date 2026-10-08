"""
Movie search routes for FastAPI
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Response

from src.schemas.movie import (
    MovieSearchRequest,
    MovieSearchResponse,
    MovieResult,
    HealthCheckResponse,
    DatabaseStatus,
)
from src.application.search_cloud import search_movies_cloud
from src.application.search_validator import verify_search_query
from src.config import API_VERSION
from src.application.database_status import (
    get_database_state,
    POPULATE_HINT,
    UNREACHABLE_HINT,
)

logger = logging.getLogger("moviefinder.api")

router = APIRouter(prefix="/api/v1", tags=["movies"])


def _parse_chroma_results(results) -> list[MovieResult]:
    """Convert a chromadb query result into a list of MovieResult."""
    movie_results = []
    if results.get("documents") and len(results["documents"]) > 0:
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

        for i, (doc, dist) in enumerate(zip(documents, distances)):
            metadata = metadatas[i] if i < len(metadatas) else None
            movie_results.append(MovieResult(
                document=doc,
                distance=float(dist),
                metadata=metadata
            ))
    return movie_results


def _guard_database_ready():
    """Raise 503 if the database cannot be reached at all."""
    status, _count = get_database_state()
    if status == DatabaseStatus.UNREACHABLE:
        logger.error("Search rejected: database unreachable")
        raise HTTPException(
            status_code=503,
            detail={"code": "database_unreachable", "message": UNREACHABLE_HINT}
        )
    return status


def _run_search(query: str, n_results: int) -> MovieSearchResponse:
    """Shared search path: validate, guard, search, log."""
    validation_result = verify_search_query(query)
    if validation_result != "OK!":
        return MovieSearchResponse(
            query=query, results=[], count=0, error=validation_result
        )

    status = _guard_database_ready()

    if status == DatabaseStatus.EMPTY:
        logger.warning("Search rejected: database is empty")
        return MovieSearchResponse(
            query=query, results=[], count=0, error=POPULATE_HINT
        )

    started = time.perf_counter()
    try:
        results = search_movies_cloud(query, n_results=n_results)
    except Exception:
        logger.exception("Search failed for query=%r", query)
        raise HTTPException(
            status_code=500,
            detail={"code": "search_failed", "message": UNREACHABLE_HINT}
        )

    movie_results = _parse_chroma_results(results)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "search query=%r n_results=%d returned=%d db=%s latency_ms=%.0f",
        query, n_results, len(movie_results), status.value, elapsed_ms
    )
    return MovieSearchResponse(
        query=query, results=movie_results, count=len(movie_results)
    )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(response: Response):
    """
    Health check endpoint.

    Reports the REAL state of the vector database:
    - `connected`: reachable and populated (HTTP 200)
    - `empty`: reachable but has no documents yet (HTTP 200 + guidance)
    - `unreachable`: cannot connect to Chroma Cloud (HTTP 503 + guidance)
    """
    status, count = get_database_state()

    if status == DatabaseStatus.CONNECTED:
        return HealthCheckResponse(
            status="healthy", version=API_VERSION,
            database_status=status, documents_count=count, message=None
        )

    if status == DatabaseStatus.EMPTY:
        return HealthCheckResponse(
            status="healthy", version=API_VERSION,
            database_status=status, documents_count=count, message=POPULATE_HINT
        )

    response.status_code = 503
    return HealthCheckResponse(
        status="degraded", version=API_VERSION,
        database_status=status, documents_count=0, message=UNREACHABLE_HINT
    )


@router.post(
    "/movies/search",
    response_model=MovieSearchResponse,
    responses={503: {"description": "Database unreachable"}}
)
async def search_movies(request: MovieSearchRequest):
    """
    Search for movies using semantic search.

    - **query**: The search query (business-validated, see search_validator)
    - **n_results**: Number of results to return (default: 5, max: 20)

    If the database is empty, returns 200 with an `error` field guiding
    the user to populate it. If unreachable, returns 503.
    """
    return _run_search(request.query, request.n_results)


@router.get(
    "/movies/search",
    response_model=MovieSearchResponse,
    responses={503: {"description": "Database unreachable"}}
)
async def search_movies_get(query: str, n_results: int = 5):
    """GET version of the search endpoint, for browsers and simple clients."""
    return _run_search(query, n_results)
