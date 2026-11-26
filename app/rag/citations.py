"""
Citation check: are the answer's citations backed by passages retrieved in this run?

The model is told to cite documents as [source: <file> › <section>]. After it
answers, we find every citation with a regular expression and compare it with
the passages that search_documents actually returned during the same run.

  verified   - the cited file and section were retrieved in this run
  unverified - the citation names something that was never retrieved (a
               made-up section, a misspelt one, or one from an earlier run)

This is plain code, no LLM, so it is fast, free and deterministic.

Known limitation: it checks that the SOURCE was retrieved, not that the
sentence is FAITHFUL to it. A citation can be verified while the sentence next
to it misquotes or overstates the passage. Checking faithfulness would need a
second model call (an "LLM judge") or a human reviewer.
"""

import re
from dataclasses import dataclass, field

# [source: returns_and_refunds.md › 3. Damaged goods › 3.1 Reporting deadlines]
#   group "source":  everything up to the first "›" (file names never contain "›")
#   group "section": the rest, up to the closing "]" (it may contain more "›")
# (?P<name>...) is a named group; [^\]›] means "any character except ] and ›".
CITATION_PATTERN = re.compile(
    r"\[source:\s*(?P<source>[^\]›]+?)\s*›\s*(?P<section>[^\]]+?)\s*\]"
)


# frozen=True makes instances immutable (like a Java record), so they can be put in a set.
@dataclass(frozen=True)
class Citation:
    """One citation found in an answer."""

    source: str
    section: str

    def text(self) -> str:
        """Returns the citation in the format the model was asked to use."""
        return f"[source: {self.source} › {self.section}]"


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


def extract_citations(answer: str) -> list[Citation]:
    """Finds every [source: … › …] citation in an answer.

    Input: the answer text.
    Output: distinct citations, in the order they first appear.
    """
    citations: list[Citation] = []
    for match in CITATION_PATTERN.finditer(answer):
        citation = Citation(
            normalize(match.group("source")), normalize(match.group("section"))
        )
        if citation not in citations:
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
            section path exactly match a passage retrieved in this run.
    """
    retrieved = retrieved_citations(steps)
    check = CitationCheck()
    for citation in extract_citations(answer):
        if citation in retrieved:
            check.verified.append(citation)
        else:
            check.unverified.append(citation)
    return check
