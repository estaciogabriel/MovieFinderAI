"""
Database state service.

Single place that knows how to check the real state of the Chroma
database. Interfaces (API health check, search guard) ask this module
instead of touching infrastructure directly.
"""

import logging

from src.infrastructure.chroma_repository import ChromaRepository
from src.schemas.movie import DatabaseStatus

logger = logging.getLogger("moviefinder.application")

POPULATE_HINT = (
    "The movie database is empty. Populate it first: "
    "place the TMDB CSVs in CSVs/ and run "
    "`.venv/bin/python -m scripts.upload_to_chroma`, then try again."
)

UNREACHABLE_HINT = (
    "Could not reach Chroma Cloud. Check CHROMA_API_KEY, CHROMA_TENANT "
    "and CHROMA_DATABASE in your .env, and your network connection."
)


def get_database_state():
    """
    Check the real state of the vector database.

    Returns:
        tuple[DatabaseStatus, int]: (status, documents_count)

    Never raises: an unreachable database is a state, not a crash.
    """
    try:
        count = ChromaRepository().count()
    except Exception:
        logger.exception("Chroma Cloud unreachable")
        return DatabaseStatus.UNREACHABLE, 0

    if count > 0:
        return DatabaseStatus.CONNECTED, count

    logger.warning("Chroma Cloud reachable but empty (0 documents)")
    return DatabaseStatus.EMPTY, 0
