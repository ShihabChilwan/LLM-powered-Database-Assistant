"""Tests for the citation check (app/rag/citations.py)."""

import pytest

from app.rag.citations import check_citations

DEADLINES = "3. Damaged, wrong or missing goods › 3.1 Reporting deadlines"

# The steps of one fake run: one search that retrieved two passages, and one SQL step.
STEPS = [
    {
        "tool": "search_documents",
        "error": None,
        "result": {
            "passages": [
                {"source": "returns_and_refunds.md", "section": DEADLINES},
                {"source": "returns_and_refunds.md", "section": "7. Exclusions"},
            ]
        },
    },
    {"tool": "run_sql_query", "error": None, "result": {"rows": []}},
]


def texts(citations: list) -> list[str]:
    """Returns each citation's text, for readable assertions."""
    result = []
    for citation in citations:
        result.append(citation.text())
    return result


def test_retrieved_citation_is_verified_and_repeats_count_once() -> None:
    answer = (
        f"48 hours [source: returns_and_refunds.md › {DEADLINES}]. "
        f"Again [Source :  returns_and_refunds.md ›   {DEADLINES} ]."
    )
    check = check_citations(answer, STEPS)

    assert texts(check.verified) == [f"[source: returns_and_refunds.md › {DEADLINES}]"]
    assert check.unverified == []
    assert check.all_verified()


def test_citation_not_retrieved_in_this_run_is_unverified() -> None:
    answer = (
        "[source: returns_and_refunds.md › 9. Secret clause] "
        f"[source: customer_service_faq.md › {DEADLINES}]"  # right section, wrong file
    )
    check = check_citations(answer, STEPS)

    assert check.verified == []
    assert texts(check.unverified) == [
        "[source: returns_and_refunds.md › 9. Secret clause]",
        f"[source: customer_service_faq.md › {DEADLINES}]",
    ]
    assert not check.all_verified()


@pytest.mark.parametrize(
    "malformed",
    [
        "[source: returns_and_refunds.md]",  # no "›"
        "[source: returns_and_refunds.md › ]",  # empty section
        "[source:  › 7. Exclusions]",  # empty file
        "[source: returns_and_refunds.md › 7. Exclusions",  # no closing "]" on the line
    ],
)
def test_malformed_citation_is_flagged_as_unverified(malformed: str) -> None:
    check = check_citations(f"Some claim {malformed}\nNext line.", STEPS)

    assert check.verified == []
    assert len(check.unverified) == 1
    assert check.unverified[0].is_malformed()
    assert check.unverified[0].text() == malformed


def test_two_sources_in_one_bracket_is_unverified() -> None:
    answer = "[source: returns_and_refunds.md › 7. Exclusions; returns_and_refunds.md › 2. Scope]"
    check = check_citations(answer, STEPS)

    assert check.verified == []
    assert len(check.unverified) == 1


def test_answer_without_citations_passes_and_plain_brackets_are_ignored() -> None:
    check = check_citations("VINET placed 5 orders [see the database].", STEPS)

    assert check.verified == []
    assert check.unverified == []
    assert check.all_verified()
