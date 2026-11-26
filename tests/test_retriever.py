"""Tests for retrieval (app/rag/retriever.py) and indexing (app/rag/ingest.py),
with fake embeddings and an in-memory ChromaDB collection."""

from pathlib import Path

import pytest

from app.llm import client
from app.rag import ingest, retriever

# Hand-made 2-D vectors, so the expected cosine similarities are easy to reason about.
FAKE_VECTORS = {
    "query": [1.0, 0.0],
    "closest": [0.95, 0.05],  # similarity about 0.999
    "middle": [0.6, 0.4],  # similarity about 0.83
    "far": [0.0, 1.0],  # similarity 0.0
}


def fake_embed(texts: list[str]) -> tuple[list[list[float]], None]:
    """Stands in for client.embed: looks each text up in FAKE_VECTORS."""
    vectors = []
    for text in texts:
        vectors.append(FAKE_VECTORS[text])
    return vectors, None


@pytest.fixture
def filled_collection(empty_collection, monkeypatch: pytest.MonkeyPatch):
    """The empty collection plus three chunks, and client.embed replaced by fake_embed."""
    monkeypatch.setattr(client, "embed", fake_embed)
    # Added in a scrambled order, so the test proves search() sorts by similarity.
    for name in ["far", "closest", "middle"]:
        empty_collection.add(
            ids=[name],
            embeddings=[FAKE_VECTORS[name]],
            documents=[f"text of {name}"],
            metadatas=[
                {"chunk_id": name, "source": f"{name}.md", "section": f"Section {name}"}
            ],
        )
    return empty_collection


def test_closest_chunk_comes_first(filled_collection) -> None:
    results = retriever.search(
        "query", k=3, min_score=-1.0, collection=filled_collection
    )

    ids = []
    for chunk in results:
        ids.append(chunk.chunk_id)
    assert ids == ["closest", "middle", "far"]
    assert results[0].source == "closest.md"
    assert results[0].section == "Section closest"
    assert results[0].text == "text of closest"
    assert results[0].score > results[1].score > results[2].score


def test_chunks_below_min_score_are_dropped(filled_collection) -> None:
    results = retriever.search(
        "query", k=3, min_score=0.5, collection=filled_collection
    )

    ids = []
    for chunk in results:
        ids.append(chunk.chunk_id)
    assert ids == ["closest", "middle"]


def test_nothing_relevant_returns_empty_list(filled_collection) -> None:
    assert (
        retriever.search("query", k=3, min_score=0.9999, collection=filled_collection)
        == []
    )


def test_distance_is_converted_to_similarity() -> None:
    assert retriever.distance_to_score(0.0) == 1.0
    assert retriever.distance_to_score(0.25) == 0.75


def test_rebuilding_the_index_replaces_it(
    chroma_client, tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "a.md").write_text(
        "# A\n\n## One\n\nFirst.\n\n## Two\n\nSecond.\n", encoding="utf-8"
    )
    calls = []

    def counting_embed(texts: list[str]) -> tuple[list[list[float]], None]:
        calls.append(len(texts))
        vectors = []
        for _text in texts:
            vectors.append([1.0, 0.0])
        return vectors, None

    monkeypatch.setattr(client, "embed", counting_embed)
    monkeypatch.setattr(ingest.config, "COLLECTION_NAME", "test_rebuild")

    first = ingest.build_index(chroma_client, folder=tmp_path)
    second = ingest.build_index(chroma_client, folder=tmp_path)

    # Two chunks ("One" and "Two"; the empty title section is carried into "One"), not doubled.
    assert first["chunks"] == second["chunks"] == 2
    assert calls == [2, 2]  # our own embed function was used both times
    chroma_client.delete_collection("test_rebuild")
