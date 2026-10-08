"""
Central configuration: reads credentials from environment variables.

Values come from the process environment (or a .env file at the project
root, loaded by python-dotenv). Never hardcode secrets in source files.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (repo root is two levels above this file)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY", "")
CHROMA_TENANT = os.environ.get(
    "CHROMA_TENANT", "abe558cc-e541-421b-a88e-a1bc72b696db"
)
CHROMA_DATABASE = os.environ.get(
    "CHROMA_DATABASE", "chroma_movieKnowledgeBase"
)

# API version: single source of truth for the FastAPI app and the
# health endpoint response.
API_VERSION = "1.2.0"
