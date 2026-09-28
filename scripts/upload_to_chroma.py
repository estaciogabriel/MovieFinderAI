"""
Upload pipeline: CSVs (TMDB) -> documents -> embeddings -> Chroma Cloud.

Populates the collection configured in src/config.py (CHROMA_API_KEY,
CHROMA_TENANT, CHROMA_DATABASE, collection "movies_docs").

Usage:
    .venv/bin/python -m scripts.upload_to_chroma [--batch-size 64] [--limit 0]

The document text format matches MovieDocumentGenerator (title, year,
genres, director, cast, rating, runtime, overview) so searches behave the
same as the original pipeline. Uses `upsert`, so re-running is idempotent.
"""

import argparse
import logging
import time

import pandas as pd

from src.infrastructure.chroma_repository import ChromaRepository
from src.infrastructure.embedder import DocumentEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger("moviefinder.upload")

MODEL_NAME = "all-mpnet-base-v2"


def safe_parse(json_str):
    if pd.isna(json_str) or json_str == "":
        return []
    try:
        import ast
        return ast.literal_eval(json_str)
    except (ValueError, SyntaxError):
        return []


def extract_names(json_list, key="name", limit=None):
    names = [item.get(key, "") for item in safe_parse(json_list) if isinstance(item, dict)]
    return names[:limit] if limit else names


def get_director(crew_json):
    for person in safe_parse(crew_json):
        if isinstance(person, dict) and person.get("job") == "Director":
            return person.get("name", "")
    return ""


def build_text(movie, cast_names, director):
    """Same format as MovieDocumentGenerator.generate_document."""
    title = movie.get("title", "Unknown")
    year = str(movie.get("release_date", ""))[:4] or "Unknown"
    overview = movie.get("overview", "") or ""

    genres = extract_names(movie.get("genres", "[]"))
    parts = [f"{title} ({year})" if year != "Unknown" else title]
    if genres:
        parts.append(f"Genres: {', '.join(genres)}")
    if director:
        parts.append(f"Director: {director}")
    if cast_names:
        parts.append(f"Cast: {', '.join(cast_names)}")

    rating = movie.get("vote_average", 0)
    votes = movie.get("vote_count", 0)
    if pd.notna(rating) and rating > 0:
        if votes >= 1000:
            parts.append(f"Rating: {rating:.1f}/10 ({votes/1000:.0f}K votes)")
        else:
            parts.append(f"Rating: {rating:.1f}/10")

    runtime = movie.get("runtime", 0)
    if pd.notna(runtime) and runtime > 0:
        parts.append(f"Runtime: {int(runtime)} min")

    if overview:
        parts.append(f"\n{overview}")

    return "\n".join(parts), genres


def iter_documents(csv_dir, limit, popular_first=False):
    """Yield (movie_id, text, metadata) for every movie with a title+overview."""
    logger.info("Loading CSVs from %s ...", csv_dir)
    movies = pd.read_csv(csv_dir / "movies_metadata.csv", low_memory=False)
    credits = pd.read_csv(csv_dir / "credits.csv")

    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
    # credits.csv has duplicate ids; keep one row per movie
    credits = credits.drop_duplicates(subset="id", keep="first")
    credits_by_id = credits.set_index("id")

    movies = movies[movies["overview"].notna() & movies["title"].notna()]
    movies = movies[movies["id"].notna()]
    if popular_first:
        movies = movies.sort_values("vote_count", ascending=False)
        logger.info("Uploading most popular movies first (vote_count desc)")
    logger.info("Movies with title+overview: %d", len(movies))

    produced = 0
    for _, movie in movies.iterrows():
        movie_id = int(movie["id"])
        try:
            credit = credits_by_id.loc[movie_id]
            cast_names = extract_names(credit.get("cast", "[]"), limit=5)
            director = get_director(credit.get("crew", "[]"))
        except KeyError:
            cast_names, director = [], ""

        text, genres = build_text(movie, cast_names, director)
        metadata = {
            "movie_id": movie_id,
            "title": str(movie.get("title", "")),
            "year": str(movie.get("release_date", ""))[:4],
            "genres": ", ".join(genres),
        }
        yield movie_id, text, metadata
        produced += 1
        if limit and produced >= limit:
            return


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", default="CSVs", help="Folder with the TMDB CSVs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Upload at most N movies (0 = all)")
    parser.add_argument("--popular-first", action="store_true",
                        help="Upload the most voted movies first")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        raise SystemExit(f"CSV folder not found: {csv_dir}")

    logger.info("Loading embedding model %s ...", MODEL_NAME)
    embedder = DocumentEmbedder(model_name=MODEL_NAME)

    collection = ChromaRepository().get_collection()
    existing = collection.count()
    logger.info("Collection 'movies_docs' currently has %d documents", existing)

    started = time.perf_counter()
    total = 0
    batch_ids, batch_docs, batch_metas = [], [], []

    for movie_id, text, metadata in iter_documents(csv_dir, args.limit, args.popular_first):
        batch_ids.append(str(movie_id))
        batch_docs.append(text)
        batch_metas.append(metadata)

        if len(batch_ids) >= args.batch_size:
            embeddings = embedder.model.encode(
                batch_docs, batch_size=args.batch_size, normalize_embeddings=True
            )
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=[e.tolist() for e in embeddings],
            )
            total += len(batch_ids)
            batch_ids, batch_docs, batch_metas = [], [], []
            elapsed = time.perf_counter() - started
            rate = total / elapsed if elapsed > 0 else 0
            logger.info(
                "Uploaded %d documents (%.1f docs/s, collection now: %d)",
                total, rate, collection.count()
            )

    if batch_ids:
        embeddings = embedder.model.encode(
            batch_docs, batch_size=len(batch_ids), normalize_embeddings=True
        )
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=[e.tolist() for e in embeddings],
        )
        total += len(batch_ids)

    logger.info(
        "Done: %d documents uploaded in %.1f minutes. Collection total: %d",
        total, (time.perf_counter() - started) / 60, collection.count()
    )


if __name__ == "__main__":
    main()
