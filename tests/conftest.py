"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def setup_paths():
    """Setup Python paths for tests"""
    # Add movies_knowledge_base to path
    movies_kb_path = str(Path(__file__).parent.parent / "movies_knowledge_base")
    if movies_kb_path not in sys.path:
        sys.path.insert(0, movies_kb_path)
    
    # Add api to path
    api_path = str(Path(__file__).parent.parent / "api")
    if api_path not in sys.path:
        sys.path.insert(0, api_path)
