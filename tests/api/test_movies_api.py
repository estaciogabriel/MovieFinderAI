"""
Integration tests for Movies API
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    from api.main import app
    return TestClient(app)


class TestMoviesAPI:
    """Integration tests for movies API endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "database_status" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data
    
    @patch('api.routes.movies.search_movies_cloud')
    def test_search_movies_post_valid(self, mock_search, client):
        """Test POST search endpoint with valid query"""
        # Setup mock
        mock_search.return_value = {
            'documents': [['Movie 1 description', 'Movie 2 description']],
            'distances': [[0.1, 0.2]],
            'metadatas': [{'id': 1}, {'id': 2}]
        }
        
        # Test
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "action movie", "n_results": 2}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "action movie"
        assert data["count"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["document"] == "Movie 1 description"
        assert data["results"][0]["distance"] == 0.1
        assert data["error"] is None
    
    @patch('api.routes.movies.search_movies_cloud')
    def test_search_movies_get_valid(self, mock_search, client):
        """Test GET search endpoint with valid query"""
        # Setup mock
        mock_search.return_value = {
            'documents': [['Movie 1 description']],
            'distances': [[0.1]],
            'metadatas': [{'id': 1}]
        }
        
        # Test
        response = client.get(
            "/api/v1/movies/search",
            params={"query": "action movie", "n_results": 1}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "action movie"
        assert data["count"] == 1
        assert len(data["results"]) == 1
    
    @patch('api.routes.movies.search_movies_cloud')
    def test_search_movies_empty_query(self, mock_search, client):
        """Test search with empty query"""
        # Test POST
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "", "n_results": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["error"] == "Your search is empty, try again!"
        
        # Test GET
        response = client.get(
            "/api/v1/movies/search",
            params={"query": "", "n_results": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["error"] == "Your search is empty, try again!"
    
    @patch('api.routes.movies.search_movies_cloud')
    def test_search_movies_too_short(self, mock_search, client):
        """Test search with too short query"""
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "ab", "n_results": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["error"] == "Your search is too short, try again!"
    
    @patch('api.routes.movies.search_movies_cloud')
    def test_search_movies_special_chars_only(self, mock_search, client):
        """Test search with only special characters"""
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "!!!", "n_results": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["error"] == "Your search has only special characters, try again!"
    
    @patch('api.routes.movies.search_movies_cloud')
    def test_search_movies_no_results(self, mock_search, client):
        """Test search with no results"""
        # Setup mock with no results
        mock_search.return_value = {
            'documents': [[]],
            'distances': [[]]
        }
        
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "nonexistent movie", "n_results": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "nonexistent movie"
        assert data["count"] == 0
        assert len(data["results"]) == 0
    
    @patch('api.routes.movies.search_movies_cloud')
    def test_search_movies_exception(self, mock_search, client):
        """Test search when exception occurs"""
        # Setup mock to raise exception
        mock_search.side_effect = Exception("Database connection failed")
        
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "action movie", "n_results": 5}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]
    
    def test_search_movies_validation_n_results(self, client):
        """Test validation of n_results parameter"""
        # Test with n_results = 0 (should fail)
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "action movie", "n_results": 0}
        )
        
        assert response.status_code == 422  # Validation error
        
        # Test with n_results > 20 (should fail)
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "action movie", "n_results": 25}
        )
        
        assert response.status_code == 422  # Validation error
        
        # Test with n_results = 20 (should pass)
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "action movie", "n_results": 20}
        )
        
        assert response.status_code == 200
    
    def test_search_movies_validation_query_length(self, client):
        """Test validation of query length"""
        # Test with query too short (less than 3 characters)
        response = client.post(
            "/api/v1/movies/search",
            json={"query": "ab", "n_results": 5}
        )
        
        # This should pass the Pydantic validation (min_length=3)
        # but fail the business validation
        assert response.status_code == 200
        data = response.json()
        assert data["error"] == "Your search is too short, try again!"
        
        # Test with query too long (more than 500 characters)
        long_query = "a" * 501
        response = client.post(
            "/api/v1/movies/search",
            json={"query": long_query, "n_results": 5}
        )
        
        # This should fail Pydantic validation
        assert response.status_code == 422


class TestAPIResponseSchemas:
    """Test API response schemas"""
    
    def test_movie_search_response_schema(self, client):
        """Test that response matches the schema"""
        with patch('api.routes.movies.search_movies_cloud') as mock_search:
            mock_search.return_value = {
                'documents': [['Test movie']],
                'distances': [[0.5]],
                'metadatas': [{'id': 1}]
            }
            
            response = client.post(
                "/api/v1/movies/search",
                json={"query": "test", "n_results": 1}
            )
            
            data = response.json()
            
            # Check required fields
            assert "query" in data
            assert "results" in data
            assert "count" in data
            assert "error" in data
            
            # Check result structure
            assert isinstance(data["results"], list)
            if len(data["results"]) > 0:
                result = data["results"][0]
                assert "document" in result
                assert "distance" in result
                assert "metadata" in result
