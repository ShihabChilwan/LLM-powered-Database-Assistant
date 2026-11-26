"""
Citation check: are the answer's citations backed by passages retrieved in this run?

The model is told to cite documents as [source: <file> › <section>]. After it
answers, we find every citation with a regular expression and compare it with
the passages that search_documents actually returned during the same run.

  verified   - the cited file and section were retrieved in this run
  unverified - the citation names something that was never retrieved (a
               made-up section, a misspelt one, or one from an earlier run),
               or it is malformed: it starts with "[source:" but cannot be
               read as a file and a section (no "›", an empty part, or no
               closing "]" on the same line)

This is plain code, no LLM, so it is fast, free and deterministic.

Known limitation: it checks that the SOURCE was retrieved, not that the
sentence is FAITHFUL to it. A citation can be verified while the sentence next
to it misquotes or overstates the passage. Checking faithfulness would need a
second model call (an "LLM judge") or a human reviewer.
"""

import re
from dataclasses import dataclass, field

# Finds where a citation starts: "[source:", ignoring case and spaces, so that
# "[Source :" is caught too. Everything after it is parsed by plain string code
# in parse_citation(), which is easier to follow than one large regex.
CITATION_START = re.compile(r"\[\s*source\s*:", re.IGNORECASE)

# Separates the file from the section: "[source: <file> › <section>]".
# File names never contain it; section paths may contain several.
SEPARATOR = "›"


# frozen=True makes instances immutable (like a Java record), so they can be put in a set.
@dataclass(frozen=True)
class Citation:
    """One citation found in an answer.

    source and section are empty for a malformed citation. raw is the text as
    written in the answer; compare=False leaves it out of == and hashing, so two
    citations are equal when their source and section are equal.
    """

    source: str
    section: str
    raw: str = field(default="", compare=False)

    def is_malformed(self) -> bool:
        """True when the citation could not be read as a file and a section."""
        return not self.source or not self.section

    def text(self) -> str:
        """Returns the citation in the standard format, or as written if malformed."""
        if self.is_malformed():
            return self.raw
        return f"[source: {self.source} {SEPARATOR} {self.section}]"


@dataclass
class CitationCheck:
    """The result of check_citations()."""

    verified: list[Citation] = field(default_factory=list)
    unverified: list[Citation] = field(default_factory=list)

    def all_verified(self) -> bool:
        """True when there is no unverified citation (also True when there are none at all)."""
        return not self.unverified


def normalize(text: str) -> str:
    """Collapses runs of whitespace into one space and trims, so spacing differences do not matter."""
    return " ".join(text.split())


def find_citation_texts(answer: str) -> list[str]:
    """Returns the raw text of every "[source: ..." in the answer, in order.

    A citation normally ends at its closing "]". If there is no "]" before the
    end of the line, the text up to the end of the line is returned, so the
    malformed citation is still reported.
    """
    texts = []
    for match in CITATION_START.finditer(answer):
        line_end = answer.find("\n", match.end())
        if line_end == -1:
            line_end = len(answer)
        closing = answer.find("]", match.end(), line_end)
        end = closing + 1 if closing != -1 else line_end
        texts.append(answer[match.start() : end])
    return texts


def parse_citation(raw: str) -> Citation:
    """Turns "[source: <file> › <section>]" into a Citation.

    Input: the raw citation text from find_citation_texts().
    Output: a Citation; a malformed one (empty source and section) when the text
            has no closing "]", no "›", or an empty file or section.
    """
    malformed = Citation("", "", raw)
    if not raw.endswith("]"):
        return malformed
    inner = raw[raw.index(":") + 1 : -1]  # between "source:" and "]"
    if SEPARATOR not in inner:
        return malformed
    source, section = inner.split(SEPARATOR, 1)  # split at the first "›" only
    citation = Citation(normalize(source), normalize(section), raw)
    return malformed if citation.is_malformed() else citation


def extract_citations(answer: str) -> list[Citation]:
    """Finds every citation in an answer, including malformed ones.

    Input: the answer text.
    Output: distinct citations, in the order they first appear. Malformed ones
            are kept apart by their raw text, so each one is reported.
    """
    citations: list[Citation] = []
    seen_malformed: list[str] = []
    for raw in find_citation_texts(answer):
        citation = parse_citation(raw)
        if citation.is_malformed():
            if raw not in seen_malformed:
                seen_malformed.append(raw)
                citations.append(citation)
        elif citation not in citations:
            citations.append(citation)
    return citations


def retrieved_citations(steps: list[dict]) -> set[Citation]:
    """Collects the (source, section) of every passage retrieved in this run.

    Input: the agent's steps; only successful search_documents steps count.
    Output: a set of Citation values.
    """
    retrieved: set[Citation] = set()
    for step in steps:
        if step["tool"] != "search_documents" or step["error"]:
            continue
        for passage in step["result"]["passages"]:
            retrieved.add(
                Citation(normalize(passage["source"]), normalize(passage["section"]))
            )
    return retrieved


def check_citations(answer: str, steps: list[dict]) -> CitationCheck:
    """Splits the answer's citations into verified and unverified.

    Input: the final answer and the steps of the same run.
    Output: a CitationCheck. A citation is verified only if its file and full
            section path exactly match a passage retrieved in this run; every
            other citation, including a malformed one, is unverified.
    """
    retrieved = retrieved_citations(steps)
    check = CitationCheck()
    for citation in extract_citations(answer):
        if not citation.is_malformed() and citation in retrieved:
            check.verified.append(citation)
        else:
            check.unverified.append(citation)
    return check
