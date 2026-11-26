"""
Retrieval: finds the policy chunks most relevant to a question.

search() embeds the question with the same EMBEDDING_MODEL used to build the
index (app/rag/ingest.py), asks ChromaDB for the nearest chunks, turns
distances into similarity scores, and drops chunks that are not similar
enough. An empty result means "the documents do not cover this", which the
model can then say instead of guessing.
"""

from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from app import config
from app.llm import client


@dataclass
class RetrievedChunk:
    """One chunk returned by search(), with its similarity score (higher = more relevant)."""

    chunk_id: str
    source: str
    section: str
    text: str
    score: float


# Opened on first use, so importing this module does not touch the disk.
_collection: Collection | None = None


def get_collection() -> Collection:
    """Returns the policy collection from CHROMA_DIR, opening it on the first call.

    Raises chromadb.errors.NotFoundError if the index has not been built yet
    (run: python -m app.rag.ingest).
    """
    global _collection  # assign the module-level variable, not a new local one
    if _collection is None:
        chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = chroma.get_collection(config.COLLECTION_NAME)
    return _collection


def distance_to_score(distance: float) -> float:
    """Converts ChromaDB's cosine distance into a cosine similarity score.

    ChromaDB returns cosine DISTANCE = 1 - cosine similarity, where 0 means
    "same direction" and larger means less similar. We flip it back to
    similarity (1 - distance) so that, everywhere else in the app, a higher
    score always means more relevant. For text embeddings it is in practice
    between 0 and 1.
    """
    return 1.0 - distance


def search(
    query: str,
    k: int = config.TOP_K,
    min_score: float = config.MIN_SCORE,
    collection: Collection | None = None,
) -> list[RetrievedChunk]:
    """Finds the chunks most similar to a query.

    Input: the query text; k = maximum number of chunks; min_score = lowest
           score kept; collection = the collection to search (tests pass an
           in-memory one; by default the index on disk is used).
    Output: at most k chunks, best first, all with score >= min_score.
            An empty list when nothing is relevant enough.
    """
    if collection is None:
        collection = get_collection()
    vectors, _stats = client.embed([query])
    result = collection.query(
        query_embeddings=vectors,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = results_to_chunks(result)
    relevant = []
    for chunk in chunks:
        if chunk.score >= min_score:
            relevant.append(chunk)
    return relevant


def results_to_chunks(result: dict) -> list[RetrievedChunk]:
    """Converts a ChromaDB query result into RetrievedChunk objects, best first.

    ChromaDB answers several queries at once, so every field is a list of
    lists; [0] picks the answers for our single query.
    """
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    chunks = []
    # zip() walks the three lists side by side, like a for loop over a shared index.
    for text, metadata, distance in zip(documents, metadatas, distances):
        chunk = RetrievedChunk(
            chunk_id=metadata["chunk_id"],
            source=metadata["source"],
            section=metadata["section"],
            text=text,
            score=distance_to_score(distance),
        )
        chunks.append(chunk)
    return chunks
