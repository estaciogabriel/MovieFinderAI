"""
Central configuration: reads credentials from environment variables.

Values come from the process environment (or a .env file at the project
root, loaded by python-dotenv). Never hardcode secrets or identifiers
in source files: missing values fail fast in ChromaRepository with a
clear message instead of silently connecting to the wrong database.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (repo root is two levels above this file)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY", "")
CHROMA_TENANT = os.environ.get("CHROMA_TENANT", "")
CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE", "")

# API version: single source of truth for the FastAPI app and the
# health endpoint response.
API_VERSION = "1.2.0"
