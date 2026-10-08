"""
Unit tests for the relevance re-ranker
"""

import pytest

from src.application.rerank import (
    parse_rating,
    parse_year,
    rerank_results,
    relevance_scores,
)


def make_doc(title, rating_text=""):
    return f"{title}\nGenres: Drama\n{rating_text}".strip()


class TestParseHelpers:
    def test_parse_rating_found(self):
        assert parse_rating("Movie\nRating: 7.5/10") == 7.5

    def test_parse_rating_with_votes(self):
        assert parse_rating("Movie\nRating: 8.2/10 (3K votes)") == 8.2

    def test_parse_rating_missing(self):
        assert parse_rating("Movie without rating") == 0.0

    def test_parse_year_from_metadata(self):
        assert parse_year({"year": "1997"}) == 1997

    def test_parse_year_empty_metadata(self):
        assert parse_year({}) == 0
        assert parse_year(None) == 0

    def test_parse_year_invalid(self):
        assert parse_year({"year": "abcd"}) == 0


class TestRelevanceScores:
    def test_newer_movie_wins_with_same_similarity(self):
        docs = [make_doc("Old", "Rating: 8.0/10"), make_doc("New", "Rating: 8.0/10")]
        metas = [{"year": "1980"}, {"year": "2020"}]
        ranked = relevance_scores(docs, [0.5, 0.5], metas)
        assert ranked[0][0] == 1  # newer first

    def test_higher_rating_wins_with_same_similarity(self):
        docs = [make_doc("Low", "Rating: 5.0/10"), make_doc("High", "Rating: 9.0/10")]
        metas = [{"year": "2000"}, {"year": "2000"}]
        ranked = relevance_scores(docs, [0.5, 0.5], metas)
        assert ranked[0][0] == 1  # higher rated first

    def test_semantic_similarity_dominates(self):
        # Very close match with a low rating must beat a distant match
        # with a perfect rating: semantic weight (0.6) > rating weight (0.2)
        docs = [make_doc("Close", "Rating: 4.0/10"), make_doc("Far", "Rating: 10/10")]
        metas = [{"year": "1990"}, {"year": "2024"}]
        ranked = relevance_scores(docs, [0.2, 1.6], metas)
        assert ranked[0][0] == 0  # the semantically close doc wins

    def test_single_result_gets_max_year_score(self):
        docs = [make_doc("Only", "Rating: 7.0/10")]
        ranked = relevance_scores(docs, [0.8], [{"year": "2000"}])
        assert len(ranked) == 1
        assert ranked[0][1] > 0


class TestRerankResults:
    def test_shapes_preserved_and_sorted(self):
        results = {
            'documents': [[make_doc("A", "Rating: 6.0/10"), make_doc("B", "Rating: 9.0/10")]],
            'distances': [[0.5, 0.6]],
            'metadatas': [[{'year': '2000', 'title': 'A'}, {'year': '2010', 'title': 'B'}]],
        }
        out = rerank_results(results)
        assert out['documents'][0][0].startswith("B")
        assert out['metadatas'][0][0]['title'] == 'B'
        assert 'relevance' in out['metadatas'][0][0]
        # original relevance order: descending
        rels = [m['relevance'] for m in out['metadatas'][0]]
        assert rels == sorted(rels, reverse=True)

    def test_empty_results_passthrough(self):
        results = {'documents': [[]], 'distances': [[]]}
        assert rerank_results(results) == results

    def test_missing_metadatas_handled(self):
        results = {
            'documents': [[make_doc("A", "Rating: 8.0/10")]],
            'distances': [[0.4]],
        }
        out = rerank_results(results)
        assert out['metadatas'][0][0]['relevance'] > 0
