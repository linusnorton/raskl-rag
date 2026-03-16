"""Pipeline step: enrich document metadata via web APIs (CrossRef, OpenLibrary, DuckDuckGo)."""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.parse
import urllib.request
import json

from ras_docproc.schema import DocumentRecord, MetadataSource

logger = logging.getLogger(__name__)

# Timeout for HTTP requests (seconds)
_HTTP_TIMEOUT = 15

# User-Agent for API requests (polite identification)
_USER_AGENT = "ras-docproc/0.1.0 (historical document processing; mailto:raskl-rag@example.com)"


def _http_get_json(url: str) -> dict | list | None:
    """Make a GET request and parse JSON response. Returns None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        logger.debug("HTTP request failed for %s: %s", url, e)
        return None


def _http_get_text(url: str) -> str | None:
    """Make a GET request and return text response. Returns None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.debug("HTTP request failed for %s: %s", url, e)
        return None


# --- CrossRef API ---


def _enrich_from_crossref(document: DocumentRecord) -> None:
    """Look up metadata via CrossRef API using DOI."""
    if not document.doi:
        return

    doi = document.doi.strip().rstrip(".")
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    data = _http_get_json(url)
    if not data or "message" not in data:
        logger.debug("CrossRef: no result for DOI %s", doi)
        return

    msg = data["message"]
    logger.info("CrossRef: found record for DOI %s", doi)

    # Title
    titles = msg.get("title", [])
    if titles and not document.title:
        document.title = titles[0]
        document.metadata_sources.append(
            MetadataSource(field="title", source="crossref", confidence=1.0, raw_value=titles[0])
        )

    # Author
    authors = msg.get("author", [])
    if authors and not document.author:
        author_parts = []
        for a in authors:
            given = a.get("given", "")
            family = a.get("family", "")
            if given and family:
                author_parts.append(f"{given} {family}")
            elif family:
                author_parts.append(family)
        if author_parts:
            author_str = "; ".join(author_parts)
            document.author = author_str
            document.metadata_sources.append(
                MetadataSource(field="author", source="crossref", confidence=1.0, raw_value=author_str)
            )

    # Editor
    editors = msg.get("editor", [])
    if editors and not document.editor:
        editor_parts = []
        for e in editors:
            given = e.get("given", "")
            family = e.get("family", "")
            if given and family:
                editor_parts.append(f"{given} {family}")
            elif family:
                editor_parts.append(family)
        if editor_parts:
            editor_str = "; ".join(editor_parts)
            document.editor = editor_str
            document.metadata_sources.append(
                MetadataSource(field="editor", source="crossref", confidence=1.0, raw_value=editor_str)
            )

    # Year
    date_parts = msg.get("published-print", msg.get("published-online", msg.get("issued", {})))
    if date_parts and not document.year:
        parts = date_parts.get("date-parts", [[]])
        if parts and parts[0] and len(parts[0]) >= 1:
            try:
                year = int(parts[0][0])
                document.year = year
                document.metadata_sources.append(
                    MetadataSource(field="year", source="crossref", confidence=1.0, raw_value=str(year))
                )
            except (ValueError, TypeError):
                pass

    # Publication (journal/container title)
    container = msg.get("container-title", [])
    if container and not document.publication:
        document.publication = container[0]
        document.metadata_sources.append(
            MetadataSource(field="publication", source="crossref", confidence=1.0, raw_value=container[0])
        )

    # Volume, issue
    if msg.get("volume") and not document.volume:
        document.volume = msg["volume"]
        document.metadata_sources.append(
            MetadataSource(field="volume", source="crossref", confidence=1.0, raw_value=msg["volume"])
        )
    if msg.get("issue") and not document.issue:
        document.issue = msg["issue"]
        document.metadata_sources.append(
            MetadataSource(field="issue", source="crossref", confidence=1.0, raw_value=msg["issue"])
        )

    # Page range
    if msg.get("page") and not document.page_range_label:
        document.page_range_label = msg["page"]
        document.metadata_sources.append(
            MetadataSource(field="page_range_label", source="crossref", confidence=1.0, raw_value=msg["page"])
        )

    # ISSN
    issns = msg.get("ISSN", [])
    if issns and not document.issn:
        document.issn = issns[0]
        document.metadata_sources.append(
            MetadataSource(field="issn", source="crossref", confidence=1.0, raw_value=issns[0])
        )

    # ISBN
    isbns = msg.get("ISBN", [])
    if isbns and not document.isbn:
        document.isbn = isbns[0]
        document.metadata_sources.append(
            MetadataSource(field="isbn", source="crossref", confidence=1.0, raw_value=isbns[0])
        )

    # Abstract
    abstract = msg.get("abstract", "")
    if abstract and not document.abstract:
        # CrossRef abstracts may contain JATS XML tags
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        document.abstract = abstract
        document.metadata_sources.append(
            MetadataSource(field="abstract", source="crossref", confidence=1.0, raw_value=abstract[:200])
        )

    # Subjects
    subjects = msg.get("subject", [])
    if subjects:
        existing = set(document.keywords)
        for s in subjects:
            if s not in existing:
                document.keywords.append(s)
                existing.add(s)
        document.metadata_sources.append(
            MetadataSource(field="keywords", source="crossref", confidence=0.9, raw_value=", ".join(subjects))
        )

    # URL
    if msg.get("URL") and not document.url:
        document.url = msg["URL"]
        document.metadata_sources.append(
            MetadataSource(field="url", source="crossref", confidence=1.0, raw_value=msg["URL"])
        )


# --- OpenLibrary API ---


def _enrich_from_openlibrary(document: DocumentRecord) -> None:
    """Search OpenLibrary by title+author for book-level metadata."""
    # Only search if we have a title
    if not document.title:
        return

    # Build search query
    q_parts = [document.title]
    if document.author:
        q_parts.append(document.author.split(";")[0].strip())

    query = " ".join(q_parts)
    url = f"https://openlibrary.org/search.json?q={urllib.parse.quote(query)}&limit=3"
    data = _http_get_json(url)
    if not data or not data.get("docs"):
        logger.debug("OpenLibrary: no results for query '%s'", query[:80])
        return

    # Take the first result
    doc = data["docs"][0]
    logger.info("OpenLibrary: found '%s' by %s", doc.get("title", "?"), doc.get("author_name", ["?"])[0])

    # ISBN
    isbns = doc.get("isbn", [])
    if isbns and not document.isbn:
        document.isbn = isbns[0]
        document.metadata_sources.append(
            MetadataSource(field="isbn", source="openlibrary", confidence=0.8, raw_value=isbns[0])
        )

    # Publisher
    publishers = doc.get("publisher", [])
    if publishers and not document.publication:
        document.publication = publishers[0]
        document.metadata_sources.append(
            MetadataSource(field="publication", source="openlibrary", confidence=0.7, raw_value=publishers[0])
        )

    # Year
    first_publish_year = doc.get("first_publish_year")
    if first_publish_year and not document.year:
        try:
            document.year = int(first_publish_year)
            document.metadata_sources.append(
                MetadataSource(
                    field="year", source="openlibrary", confidence=0.7, raw_value=str(first_publish_year)
                )
            )
        except (ValueError, TypeError):
            pass

    # Subjects as keywords
    subjects = doc.get("subject", [])[:10]
    if subjects:
        existing = set(document.keywords)
        added = []
        for s in subjects:
            if s not in existing:
                document.keywords.append(s)
                existing.add(s)
                added.append(s)
        if added:
            document.metadata_sources.append(
                MetadataSource(field="keywords", source="openlibrary", confidence=0.7, raw_value=", ".join(added))
            )

    # Editor
    if not document.editor:
        # OpenLibrary doesn't have a direct editor field in search, skip
        pass


# --- DuckDuckGo Instant Answer API ---


def _enrich_from_duckduckgo(document: DocumentRecord) -> None:
    """Search DuckDuckGo Instant Answer API for additional context.

    Uses the free Instant Answer API to find Wikipedia summaries and related info.
    """
    if not document.title:
        return

    # Build search query focused on the document
    q_parts = [document.title]
    if document.author:
        q_parts.append(document.author.split(";")[0].strip())

    query = " ".join(q_parts)
    url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
    data = _http_get_json(url)
    if not data:
        return

    # Abstract from DuckDuckGo (usually Wikipedia)
    abstract_text = data.get("AbstractText", "")
    abstract_source = data.get("AbstractSource", "")
    if abstract_text and not document.description:
        # Use as description since DDG abstracts are about the topic, not the document itself
        document.description = abstract_text[:500]
        document.metadata_sources.append(
            MetadataSource(
                field="description",
                source=f"duckduckgo/{abstract_source}",
                confidence=0.5,
                raw_value=abstract_text[:200],
            )
        )

    # Related topics can provide keywords
    related = data.get("RelatedTopics", [])
    if related and len(document.keywords) < 5:
        existing = set(document.keywords)
        for topic in related[:5]:
            text = topic.get("Text", "")
            if text and len(text) > 10:
                # Extract the first sentence as a potential keyword/topic
                first_sentence = text.split(".")[0].strip()
                if first_sentence and len(first_sentence) < 80 and first_sentence not in existing:
                    document.keywords.append(first_sentence)
                    existing.add(first_sentence)
                    break  # Just take 1 related topic to avoid noise


# --- Main enrichment function ---


def enrich_metadata_web(document: DocumentRecord) -> DocumentRecord:
    """Enrich document metadata using web APIs.

    Calls CrossRef (by DOI), OpenLibrary (by title/author), and DuckDuckGo
    in sequence. Each source only fills fields that are currently None/empty.

    Priority: CrossRef > OpenLibrary > DuckDuckGo (by confidence).

    Mutates and returns the DocumentRecord.
    """
    # 1. CrossRef (highest quality — authoritative bibliographic data by DOI)
    try:
        _enrich_from_crossref(document)
    except Exception:
        logger.warning("CrossRef enrichment failed", exc_info=True)

    # 2. OpenLibrary (good for books, ISBNs, subjects)
    try:
        _enrich_from_openlibrary(document)
    except Exception:
        logger.warning("OpenLibrary enrichment failed", exc_info=True)

    # 3. DuckDuckGo (contextual info, Wikipedia summaries)
    try:
        _enrich_from_duckduckgo(document)
    except Exception:
        logger.warning("DuckDuckGo enrichment failed", exc_info=True)

    sources_used = {s.source.split("/")[0] for s in document.metadata_sources}
    logger.info("Web enrichment complete. Sources used: %s", ", ".join(sorted(sources_used)) or "none")
    return document
