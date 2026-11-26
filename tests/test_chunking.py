"""Tests for the Markdown chunker (app/rag/chunking.py)."""

from app import config
from app.rag.chunking import chunk_document

SHORT_DOC = """Synthetic sample document created for this project.

# Test Policy

Intro line.

## 1. Returns

Goods can be returned within 30 days.

## 2. Damage

### 2.1 Deadlines

Report damage within 48 hours.
"""


def long_section_doc() -> str:
    """A document with one section of 12 paragraphs, about 2,000 characters."""
    paragraphs = []
    for index in range(12):
        paragraphs.append(f"Paragraph {index}: " + "words about the policy " * 7)
    return "# Doc\n\n## Long section\n\n" + "\n\n".join(paragraphs) + "\n"


def big_table_doc() -> str:
    """A document with one table of 40 rows (well over CHUNK_SIZE characters)."""
    rows = ["| Product | Rule |", "|---|---|"]
    for index in range(40):
        rows.append(f"| Product number {index} | Store at 2 to 5 degrees |")
    return (
        "# Doc\n\n## Storage\n\nIntro text.\n\n"
        + "\n".join(rows)
        + "\n\nAfter the table.\n"
    )


def test_splits_by_heading_and_keeps_heading_path() -> None:
    chunks = chunk_document("test.md", SHORT_DOC)

    sections = []
    for chunk in chunks:
        sections.append(chunk.section)
    assert sections == ["Test Policy", "1. Returns", "2. Damage › 2.1 Deadlines"]


def test_every_chunk_starts_with_its_heading_and_keeps_its_source() -> None:
    chunks = chunk_document("test.md", SHORT_DOC)

    # The text before the first heading joins the first chunk.
    assert chunks[0].text.startswith("Synthetic sample document")
    assert chunks[1].text.startswith("## 1. Returns")
    # A heading with no text of its own ("## 2. Damage") is carried into the next chunk.
    assert chunks[2].text.startswith("## 2. Damage\n\n### 2.1 Deadlines")
    for chunk in chunks:
        assert chunk.source == "test.md"
        assert chunk.chunk_id.startswith("test.md#")


def test_long_section_is_split_within_size_with_overlap() -> None:
    chunks = chunk_document("long.md", long_section_doc())

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= config.CHUNK_SIZE
        assert chunk.section == "Long section"
        # The empty "# Doc" heading is carried down, so every piece repeats both headings.
        assert chunk.text.startswith("# Doc\n\n## Long section")
    # Every piece after the first starts (after its heading) with overlap from the previous one.
    for chunk in chunks[1:]:
        assert "\n\n..." in chunk.text


def test_table_is_never_split_even_when_too_large() -> None:
    chunks = chunk_document("table.md", big_table_doc())

    table_chunks = []
    for chunk in chunks:
        if "| Product |" in chunk.text:
            table_chunks.append(chunk)
    assert len(table_chunks) == 1
    table_chunk = table_chunks[0]
    assert len(table_chunk.text) > config.CHUNK_SIZE
    assert "| Product number 0 |" in table_chunk.text
    assert "| Product number 39 |" in table_chunk.text
    # No other chunk holds a stray table row.
    for chunk in chunks:
        if chunk is not table_chunk:
            assert "| Product number" not in chunk.text
