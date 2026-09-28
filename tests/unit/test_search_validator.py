"""
Unit tests for search validator
"""

import pytest

from src.application.search_validator import verify_search_query


class TestSearchValidator:
    """Test cases for search query validation"""
    
    def test_valid_query(self):
        """Test valid search queries"""
        valid_queries = [
            "action movie",
            "sci-fi film",
            "comedy with Tom Hanks",
            "12345",
            "a b c",
            "The Dark Knight",
        ]
        
        for query in valid_queries:
            result = verify_search_query(query)
            assert result == "OK!", f"Query '{query}' should be valid"
    
    def test_empty_query(self):
        """Test empty query"""
        empty_queries = ["", "   ", "\t", "\n"]
        
        for query in empty_queries:
            result = verify_search_query(query)
            assert result == "Your search is empty, try again!"
    
    def test_too_short_query(self):
        """Test queries that are too short"""
        short_queries = ["ab", "a", "12", "x"]
        
        for query in short_queries:
            result = verify_search_query(query)
            assert result == "Your search is too short, try again!"
    
    def test_special_characters_only(self):
        """Test queries with only special characters"""
        # Note: whitespace-only queries are caught by empty check first
        special_only = ["!!!", "???", "...", "@@@", "###"]
        
        for query in special_only:
            result = verify_search_query(query)
            assert result == "Your search has only special characters, try again!"
    
    def test_mixed_special_and_valid(self):
        """Test queries with special characters and valid text"""
        mixed_queries = [
            "hello!",
            "what?",
            "movie...",
            "action!!!",
            "a!b@c#",
        ]
        
        for query in mixed_queries:
            result = verify_search_query(query)
            assert result == "OK!", f"Query '{query}' should be valid"
    
    def test_whitespace_variations(self):
        """Test queries with various whitespace patterns"""
        # Should be valid (has alphanumeric characters)
        result = verify_search_query("  hello world  ")
        assert result == "OK!"
        
        # Should be invalid (only whitespace)
        result = verify_search_query("   ")
        assert result == "Your search is empty, try again!"
    
    def test_unicode_characters(self):
        """Test queries with unicode characters"""
        unicode_queries = [
            "café",
            "naïve",
            "日本語",
            "emoji 🎬",
        ]
        
        for query in unicode_queries:
            result = verify_search_query(query)
            assert result == "OK!", f"Query '{query}' should be valid"
