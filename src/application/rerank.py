"""
Relevance re-ranking for movie search results.

Semantic distance alone ranks by meaning similarity. This module re-ranks
the semantic candidates by combining three signals:

    relevance = W_SEMANTIC * similarity
              + W_RATING  * rating_score
              + W_YEAR    * year_score

- similarity:   1 - distance^2/2. The collection uses Chroma's default
                L2 metric on normalized embeddings, where L2^2 = 2 - 2*cos,
                so this recovers the exact cosine similarity.
- rating_score: vote_average / 10 (parsed from the document text)
- year_score:   min-max normalized release year within the result set
                (newer = more relevant)

Weights are tuned so semantic similarity still dominates: the re-ranker
shuffles semantic neighbors, it does not replace the search.
"""

import re

RATING_RE = re.compile(r"Rating:\s*([\d.]+)/10")

W_SEMANTIC = 0.6
W_RATING = 0.2
W_YEAR = 0.2


def parse_rating(document: str) -> float:
    """Extract the TMDB rating (0-10) from the document text, if present."""
    match = RATING_RE.search(document)
    return float(match.group(1)) if match else 0.0


def parse_year(metadata) -> int:
    """Extract the release year from the document metadata, if present."""
    if not metadata:
        return 0
    try:
        return int(str(metadata.get("year", ""))[:4])
    except (ValueError, TypeError):
        return 0


def relevance_scores(documents, distances, metadatas):
    """
    Compute a relevance score for each (document, distance, metadata).

    Returns a list of (index, score) sorted from most to least relevant.
    """
    if not documents:
        return []

    years = [parse_year(m) for m in metadatas] or [0] * len(documents)
    year_min, year_max = min(years), max(years)

    scored = []
    for i, (doc, dist) in enumerate(zip(documents, distances)):
        similarity = 1.0 - (dist ** 2) / 2.0    # L2^2 = 2 - 2*cos
        rating_score = parse_rating(doc) / 10.0  # in [0, 1]
        if year_max > year_min:
            year_score = (years[i] - year_min) / (year_max - year_min)
        else:
            year_score = 1.0

        score = (
            W_SEMANTIC * similarity
            + W_RATING * rating_score
            + W_YEAR * year_score
        )
        scored.append((i, score))
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def rerank_results(results: dict) -> dict:
    """
    Re-rank a chromadb query result by relevance (semantic + rating + year).

    Accepts and returns the same chroma result shape:
    {"documents": [[...]], "distances": [[...]], "metadatas": [[...]]}
    The relevance score is injected into each item's metadata.
    """
    if not results or not results.get("documents") or not results["documents"][0]:
        return results

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas") or [[] for _ in documents]
    metadatas = metadatas[0] if metadatas else []

    ranked = relevance_scores(documents, distances, metadatas)

    new_documents, new_distances, new_metadatas = [], [], []
    for i, score in ranked:
        new_documents.append(documents[i])
        new_distances.append(distances[i])
        metadata = dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {}
        metadata["relevance"] = round(score, 4)
        new_metadatas.append(metadata)

    return {
        "documents": [new_documents],
        "distances": [new_distances],
        "metadatas": [new_metadatas],
    }
