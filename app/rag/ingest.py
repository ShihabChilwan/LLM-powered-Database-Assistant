"""
Builds the document index: policy Markdown files -> chunks -> embeddings -> ChromaDB.

Run from the project root with:  python -m app.rag.ingest

What ChromaDB does for us
-------------------------
ChromaDB is an embedded vector database: it runs inside this Python process
and keeps its files in app/rag/chroma_db/, so there is no server to start
(similar to SQLite, but for vectors).

It stores each chunk's vector together with its text and metadata (source,
section, chunk_id). When a question comes in, we give it the question's vector
and it returns the nearest chunks. To do that quickly it keeps an approximate
nearest-neighbour index called HNSW (Hierarchical Navigable Small World): a
layered graph that links each vector to a few similar ones. A search walks
that graph towards the question's vector instead of comparing the question
with every chunk one by one. The answer is "approximate" because the walk can
in rare cases miss the true nearest chunk, a trade-off for speed that only
matters with far more chunks than we have.

Why we make the embeddings ourselves
------------------------------------
ChromaDB can embed text with its own built-in default model. We do not use
it: we call client.embed() (EMBEDDING_MODEL through OpenRouter) instead, so we
control which model makes the vectors. Vectors from different models are not
comparable, so the same model must embed both the documents (here) and the
questions (in the retriever).
"""

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from app import config
from app.llm import client
from app.rag.chunking import Chunk, chunk_document


def load_documents(folder: Path) -> list[tuple[str, str]]:
    """Reads every .md file in a folder.

    Input: the folder path.
    Output: (file name, text) pairs, sorted by file name so the build is repeatable.
    """
    documents = []
    for path in sorted(folder.glob("*.md")):
        documents.append((path.name, path.read_text(encoding="utf-8")))
    return documents


def build_chunks(documents: list[tuple[str, str]]) -> list[Chunk]:
    """Chunks every document. Input: (file name, text) pairs. Output: all chunks, in order."""
    chunks: list[Chunk] = []
    for source, text in documents:
        chunks.extend(chunk_document(source, text))
    return chunks


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Embeds the text of every chunk, EMBED_BATCH_SIZE chunks per request.

    Input: the chunks.
    Output: one vector per chunk, in the same order.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), config.EMBED_BATCH_SIZE):
        batch = chunks[
            start : start + config.EMBED_BATCH_SIZE
        ]  # a slice, like List.subList
        texts = []
        for chunk in batch:
            texts.append(chunk.text)
        batch_vectors, _stats = client.embed(texts)
        vectors.extend(batch_vectors)
    return vectors


def recreate_collection(chroma: chromadb.ClientAPI) -> Collection:
    """Deletes the policy collection if it exists and creates an empty one.

    Rebuilding from empty means the index never mixes chunks from an old
    version of the documents with chunks from the new one.
    """
    try:
        chroma.delete_collection(config.COLLECTION_NAME)
    except NotFoundError:
        pass  # first build: nothing to delete
    return chroma.create_collection(
        name=config.COLLECTION_NAME,
        # Cosine distance compares the direction of vectors, not their length,
        # which is the usual choice for text embeddings. Chroma's default is l2.
        configuration={"hnsw": {"space": "cosine"}},
        # None = no built-in embedding model; we always pass our own vectors.
        embedding_function=None,
    )


def store_chunks(
    collection: Collection, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    """Adds the chunks, their vectors and their metadata to the collection."""
    ids = []
    documents = []
    metadatas = []
    for chunk in chunks:
        ids.append(chunk.chunk_id)
        documents.append(chunk.text)
        metadatas.append(
            {
                "source": chunk.source,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
            }
        )
    collection.add(
        ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas
    )


def build_index(chroma: chromadb.ClientAPI, folder: Path = config.POLICIES_DIR) -> dict:
    """Runs the whole build: load, chunk, embed, store.

    Input: a ChromaDB client (persistent here; tests can pass an in-memory one)
           and the folder of documents.
    Output: a summary dict with the document count, chunk count and chunks per document.
    """
    documents = load_documents(folder)
    chunks = build_chunks(documents)
    vectors = embed_chunks(chunks)
    collection = recreate_collection(chroma)
    store_chunks(collection, chunks, vectors)

    chunks_per_document: dict[str, int] = {}
    for chunk in chunks:
        # dict.get(key, 0) returns 0 when the key is missing, like Map.getOrDefault.
        chunks_per_document[chunk.source] = chunks_per_document.get(chunk.source, 0) + 1
    largest = max(chunks, key=lambda chunk: len(chunk.text))
    return {
        "documents": len(documents),
        "chunks": collection.count(),
        "chunks_per_document": chunks_per_document,
        "largest_chunk": (largest.chunk_id, len(largest.text)),
    }


def main() -> None:
    """Builds the index on disk at CHROMA_DIR and prints a summary."""
    chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    summary = build_index(chroma)
    print(f"Embedding model: {config.EMBEDDING_MODEL}")
    print(f"Documents: {summary['documents']}")
    print(f"Chunks: {summary['chunks']}")
    for source, count in summary["chunks_per_document"].items():
        print(f"  {source}: {count}")
    largest_id, largest_size = summary["largest_chunk"]
    print(f"Largest chunk: {largest_id} ({largest_size} characters)")
    print(
        f"Index stored in {config.CHROMA_DIR} (collection '{config.COLLECTION_NAME}')"
    )


if __name__ == "__main__":
    main()
