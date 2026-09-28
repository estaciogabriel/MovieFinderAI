"""
Pydantic schemas for MovieFinderAI API

These models are the API contract: every endpoint's request and response
shape is defined here. Frontends and integrations can rely on them.
"""

from enum import Enum

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DatabaseStatus(str, Enum):
    """Real state of the vector database, as seen by the API."""
    CONNECTED = "connected"      # reachable and has documents
    EMPTY = "empty"              # reachable, but zero documents
    UNREACHABLE = "unreachable"  # cannot connect (credentials, network, outage)


class MovieSearchRequest(BaseModel):
    """Request schema for movie search"""
    query: str = Field(..., description="Search query for movies", max_length=500)
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class MovieResult(BaseModel):
    """Schema for a single movie search result"""
    document: str = Field(..., description="Movie document/description")
    distance: float = Field(..., description="Distance score from query")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class MovieSearchResponse(BaseModel):
    """Response schema for movie search"""
    query: str = Field(..., description="Original search query")
    results: List[MovieResult] = Field(default_factory=list, description="List of movie results")
    count: int = Field(..., description="Total number of results")
    error: Optional[str] = Field(default=None, description="Error message if any")


class HealthCheckResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="API status: 'healthy' or 'degraded'")
    version: str = Field(..., description="API version")
    database_status: DatabaseStatus = Field(..., description="Real state of the vector database")
    documents_count: int = Field(..., description="Number of documents currently in the database")
    message: Optional[str] = Field(
        default=None,
        description="Guidance for the user when something needs attention"
    )


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Error details")


class ValidationErrorResponse(BaseModel):
    """Response schema for validation errors"""
    error: str = Field(default="Validation error", description="Error type")
    details: List[Dict[str, Any]] = Field(default_factory=list, description="Validation error details")
