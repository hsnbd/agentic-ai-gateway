"""Text extraction for RAG uploads: text, Markdown, HTML, PDF, and Word."""

from __future__ import annotations

import pytest

from app.core.errors import InvalidRequestError
from app.rag.extract import extract_text, html_to_text
from tests.support.documents import make_docx, make_pdf


def test_plain_text_and_markdown() -> None:
    assert extract_text(b"hello", "a.txt") == ("hello", "text/plain")
    assert extract_text(b"# Title", "README.MD") == ("# Title", "text/markdown")


def test_html_keeps_visible_text_in_blocks() -> None:
    html = (
        "<html><head><title>x</title><style>p{}</style></head><body>"
        "<h1>Refunds</h1><script>track()</script><p>Within  14 days.</p>"
        "<ul><li>Unused &amp; boxed</li></ul></body></html>"
    )
    assert html_to_text(html) == "Refunds\nWithin 14 days.\nUnused & boxed"
    assert extract_text(html.encode(), "page.htm")[1] == "text/html"


def test_pdf_text_layer_is_extracted() -> None:
    text, media_type = extract_text(make_pdf("Error E-4471 means a postcode mismatch"), "e.pdf")
    assert "E-4471" in text and media_type == "application/pdf"


def test_docx_paragraphs_and_tables_are_extracted() -> None:
    data = make_docx(["Travel policy", "Economy only."], table=[["Meal", "60 USD"]])
    text, _ = extract_text(data, "policy.docx")
    assert text == "Travel policy\nEconomy only.\nMeal | 60 USD"


@pytest.mark.parametrize(
    ("data", "name", "status", "message"),
    [
        (b"MZ", "tool.exe", 415, "Cannot ingest .exe"),
        (b"data", "Makefile", 415, "files without an extension"),
        (b"%PDF-broken", "x.pdf", 422, "Could not read the PDF"),
        (b"not a zip", "x.docx", 422, "Could not read the Word file"),
        (b"   \n", "blank.txt", 422, "No text could be extracted"),
    ],
)
def test_unsupported_unreadable_and_empty_files_are_refused(
    data: bytes, name: str, status: int, message: str
) -> None:
    with pytest.raises(InvalidRequestError, match=message) as refused:
        extract_text(data, name)
    assert refused.value.status_code == status
