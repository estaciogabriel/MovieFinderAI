"""
Main FastAPI application for MovieFinderAI
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.interfaces.api.routes.movies import router as movies_router

# Application logging: timestamped, level-tagged, one line per event.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("moviefinder.api")

# Create FastAPI app
app = FastAPI(
    title="MovieFinderAI API",
    description="Semantic search API for movies using embeddings and Chroma Cloud",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(movies_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MovieFinderAI API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
