"""
Splits a Markdown document into chunks for the document index.

Rules, in order:
  1. Split by heading: every heading starts a new section. A heading that has
     no text of its own (it is directly followed by a sub-heading) is carried
     down and printed above the next section's text, so no heading is lost.
  2. Each chunk starts with its section's heading line(s), so a chunk always
     says what it is about, even when read on its own.
  3. A section longer than CHUNK_SIZE is split between paragraphs. Each later
     piece starts with about CHUNK_OVERLAP characters from the end of the
     previous piece.
  4. A Markdown table is never split, even if it is longer than CHUNK_SIZE:
     half a table, without its header row, is meaningless.

This module is plain text processing: no embeddings, no database. That keeps
it easy to test.
"""

import re
from dataclasses import dataclass

from app import config

# A Markdown heading: 1 to 6 '#' characters, a space, then the title.
# re.compile builds the pattern once, like java.util.regex.Pattern.compile.
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Joins heading titles into a section path, e.g. "3. Damaged goods › 3.1 Deadlines".
SECTION_SEPARATOR = " › "

# Marks overlap text copied from the previous chunk.
OVERLAP_MARKER = "..."


@dataclass
class RawSection:
    """One heading and the lines below it, before any processing."""

    level: int  # 1 for '#', 2 for '##', ...
    title: str
    heading_line: str
    body: str


@dataclass
class Section:
    """A heading path, the heading text printed above the body, and the body."""

    path: str
    heading_text: str
    body: str


@dataclass
class Block:
    """A paragraph, list or table: the smallest unit the splitter moves around."""

    text: str
    is_table: bool


@dataclass
class Chunk:
    """One piece of a document, as stored in the index."""

    chunk_id: str
    source: str  # file name, e.g. "returns_and_refunds.md"
    section: str  # heading path, e.g. "3. Damaged, wrong or missing goods › 3.1 Reporting deadlines"
    text: str


# ---------- Step 1: headings and sections ----------


def parse_headings(markdown: str) -> tuple[str, list[RawSection]]:
    """Splits a document at every heading line.

    Input: the whole Markdown text.
    Output: (preamble, raw_sections). The preamble is any text before the first
            heading; each RawSection holds one heading and the text below it.
    """
    preamble_lines: list[str] = []
    raw_sections: list[RawSection] = []
    for line in markdown.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            level = len(match.group(1))
            raw_sections.append(RawSection(level, match.group(2), line, ""))
        elif raw_sections:
            raw_sections[-1].body += line + "\n"
        else:
            preamble_lines.append(line)
    return "\n".join(preamble_lines).strip(), raw_sections


def section_path(titles_by_level: dict[int, str]) -> str:
    """Builds the heading path from the titles of the currently open headings.

    The document title (level 1) is left out when there are deeper headings,
    because the source file name already identifies the document.
    """
    levels = sorted(titles_by_level)
    if len(levels) > 1 and levels[0] == 1:
        levels = levels[1:]
    titles = []
    for level in levels:
        titles.append(titles_by_level[level])
    return SECTION_SEPARATOR.join(titles)


def build_sections(preamble: str, raw_sections: list[RawSection]) -> list[Section]:
    """Turns raw sections into sections with a heading path.

    Headings with an empty body (and the preamble) are carried down and shown
    above the next section that has text, so each heading stays with content.
    """
    sections: list[Section] = []
    titles_by_level: dict[int, str] = {}
    carried_lines = [preamble] if preamble else []
    for raw in raw_sections:
        # Opening a heading closes every heading at the same or a deeper level.
        for level in list(titles_by_level):
            if level >= raw.level:
                del titles_by_level[level]
        titles_by_level[raw.level] = raw.title

        heading_lines = carried_lines + [raw.heading_line]
        if not raw.body.strip():
            carried_lines = heading_lines
            continue
        heading_text = "\n\n".join(heading_lines)
        sections.append(
            Section(section_path(titles_by_level), heading_text, raw.body.strip())
        )
        carried_lines = []
    return sections


# ---------- Step 2: blocks (paragraphs, lists, tables) ----------


def split_into_blocks(body: str) -> list[Block]:
    """Splits a section body into blocks at blank lines and at table edges.

    Consecutive lines starting with '|' form one table block, so a table is
    always kept together.
    """
    blocks: list[Block] = []
    current_lines: list[str] = []
    current_is_table = False
    for line in body.splitlines():
        is_table_line = line.lstrip().startswith("|")
        if not line.strip() or is_table_line != current_is_table:
            if current_lines:
                blocks.append(Block("\n".join(current_lines), current_is_table))
            current_lines = []
        if line.strip():
            current_lines.append(line)
            current_is_table = is_table_line
    if current_lines:
        blocks.append(Block("\n".join(current_lines), current_is_table))
    return blocks


def split_long_block(block: Block, max_chars: int) -> list[Block]:
    """Splits a non-table block that is longer than max_chars at line breaks.

    Tables and short blocks are returned unchanged. A single line longer than
    max_chars is kept whole rather than cut mid-sentence.
    """
    if block.is_table or len(block.text) <= max_chars:
        return [block]
    pieces: list[Block] = []
    current = ""
    for line in block.text.splitlines():
        if current and len(current) + 1 + len(line) > max_chars:
            pieces.append(Block(current, False))
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(Block(current, False))
    return pieces


# ---------- Step 3: pack blocks into chunks ----------


def pack_blocks(blocks: list[Block], max_chars: int) -> list[list[Block]]:
    """Groups consecutive blocks so each group's text stays within max_chars.

    A block that is too big on its own (a large table) gets a group to itself.
    """
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_length = 0
    for block in blocks:
        added_length = len(block.text) + (
            2 if current else 0
        )  # 2 = the "\n\n" between blocks
        if current and current_length + added_length > max_chars:
            groups.append(current)
            current = []
            current_length = 0
            added_length = len(block.text)
        current.append(block)
        current_length += added_length
    if current:
        groups.append(current)
    return groups


def overlap_text(previous_group: list[Block], max_chars: int) -> str:
    """Returns the last max_chars characters of the previous group, starting at a word.

    Returns "" when the previous group ends with a table, because a fragment of
    a table cannot be read on its own.
    """
    last_block = previous_group[-1]
    if last_block.is_table:
        return ""
    text = last_block.text
    if len(text) > max_chars:
        text = text[-max_chars:]
        first_space = text.find(" ")
        if first_space != -1:
            text = text[first_space + 1 :]  # start at the next whole word
    return OVERLAP_MARKER + text


def section_texts(section: Section) -> list[str]:
    """Splits one section into chunk texts, each starting with the section's heading.

    Input: a Section.
    Output: a list of chunk texts.
    """
    whole_section = f"{section.heading_text}\n\n{section.body}"
    if len(whole_section) <= config.CHUNK_SIZE:
        return [whole_section]

    # Leave room for the heading and the overlap, so ordinary chunks stay within CHUNK_SIZE.
    reserved = (
        len(section.heading_text) + config.CHUNK_OVERLAP + len(OVERLAP_MARKER) + 4
    )
    budget = max(config.CHUNK_SIZE - reserved, config.CHUNK_OVERLAP)

    blocks: list[Block] = []
    for block in split_into_blocks(section.body):
        blocks.extend(split_long_block(block, budget))

    texts: list[str] = []
    previous_group: list[Block] | None = None
    for group in pack_blocks(blocks, budget):
        parts = [section.heading_text]
        if previous_group is not None:
            overlap = overlap_text(previous_group, config.CHUNK_OVERLAP)
            if overlap:
                parts.append(overlap)
        for block in group:
            parts.append(block.text)
        texts.append("\n\n".join(parts))
        previous_group = group
    return texts


def chunk_document(source: str, markdown: str) -> list[Chunk]:
    """Splits one Markdown document into chunks.

    Input: the file name (used in chunk ids and citations) and the document text.
    Output: chunks in document order, with ids like "returns_and_refunds.md#03".
    """
    preamble, raw_sections = parse_headings(markdown)
    chunks: list[Chunk] = []
    for section in build_sections(preamble, raw_sections):
        for text in section_texts(section):
            chunk_id = (
                f"{source}#{len(chunks) + 1:02d}"  # :02d pads with zeros: 1 -> "01"
            )
            chunks.append(Chunk(chunk_id, source, section.path, text))
    return chunks
