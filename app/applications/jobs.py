"""Load job records and retrieve public employer descriptions for applications."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import ipaddress
import json
from pathlib import Path
import re
import socket
import time
from typing import Any
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import urllib3

from app.scraper.cleaner import JobCleaner
from app.scraper.deduplicator import JobDeduplicator
from app.scraper.sources.base import RawJobPage
from app.scraper.sources.dwp import DwpSource
from app.scraper.sources.generic_jsonld import GenericJsonLdSource

MAX_PAGE_BYTES = 2_000_000
MAX_DESCRIPTION_BYTES = 250_000
MAX_JOB_BYTES = 2_000_000
MAX_REDIRECTS = 5
FETCH_DEADLINE_SECONDS = 45
MIN_DESCRIPTION_CHARS = 400


class JobInputError(ValueError):
    """A job file, description or public job URL cannot be used."""


def _validate_job(job: Any) -> dict:
    if not isinstance(job, dict):
        raise JobInputError("Expected a JSON object containing one job.")
    for field in ("title", "description"):
        if not isinstance(job.get(field), str) or not job[field].strip():
            raise JobInputError(f"Job {field} must be non-empty text.")
    source_url = job.get("source_url") or job.get("url")
    _parse_url(source_url)
    if job.get("metadata") is not None and not isinstance(job["metadata"], dict):
        raise JobInputError("Job metadata must be an object or null.")
    result = deepcopy(job)
    if not result.get("source_url"):
        result["source_url"] = source_url
    return result


def load_jobs(input_dir: Path, job_files: list[Path] | None = None) -> list[dict]:
    """Load individual stored job objects, reporting every invalid file by name.

    An explicit list takes precedence over the directory. Invalid input raises
    one error listing the affected files rather than silently dropping jobs.
    """
    paths = list(job_files) if job_files is not None else sorted(Path(input_dir).glob("*.json"))
    if not paths:
        raise JobInputError("No job JSON files were found. Fetch jobs or supply --job-file.")
    jobs = []
    errors = []
    for path in paths:
        path = Path(path)
        try:
            if path.suffix.lower() != ".json":
                raise JobInputError("Expected a .json job file.")
            if path.stat().st_size > MAX_JOB_BYTES:
                raise JobInputError("Job JSON exceeds the 2 MB limit.")
            jobs.append(_validate_job(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, UnicodeError, ValueError) as error:
            detail = str(error) if isinstance(error, JobInputError) else "Cannot read valid UTF-8 job JSON."
            errors.append(f"{path.name}: {detail}")
    if errors:
        raise JobInputError("Invalid job files: " + "; ".join(errors))
    return jobs


def _parse_url(url: Any):
    if not isinstance(url, str) or not url or len(url) > 8192:
        raise JobInputError("Job URL must be an absolute HTTP or HTTPS URL.")
    if any(character.isspace() or ord(character) < 32 for character in url) or "\\" in url:
        raise JobInputError("Job URL contains invalid characters.")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise JobInputError("Job URL has an invalid host or port.") from None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise JobInputError("Job URL must be an absolute HTTP or HTTPS URL.")
    if parts.username is not None or parts.password is not None:
        raise JobInputError("Job URL must not contain credentials.")
    if port is not None and port not in {80, 443}:
        raise JobInputError("Job URLs may only use public web ports 80 and 443.")
    if "%" in parts.hostname:
        raise JobInputError("Job URL has an invalid host.")
    hostname = parts.hostname.casefold().rstrip(".")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise JobInputError("Job URL must resolve only to public Internet addresses.")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None and (not literal.is_global or literal.is_multicast):
        raise JobInputError("Job URL must resolve only to public Internet addresses.")
    return parts


def _public_destination(url: str):
    parts = _parse_url(url)
    try:
        hostname = parts.hostname.encode("idna").decode("ascii").rstrip(".")
    except UnicodeError:
        raise JobInputError("Job URL has an invalid hostname.") from None
    if hostname.casefold() == "localhost" or hostname.casefold().endswith((".localhost", ".local", ".internal")):
        raise JobInputError("Job URL must resolve only to public Internet addresses.")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        answers = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError:
        raise JobInputError("Could not resolve the public job hostname.") from None
    addresses = []
    for answer in answers:
        try:
            address = ipaddress.ip_address(answer[4][0])
        except ValueError:
            raise JobInputError("Job hostname resolved to an invalid address.") from None
        embedded = (address.ipv4_mapped or address.sixtofour) if address.version == 6 else None
        if not address.is_global or address.is_multicast or address.is_unspecified or embedded is not None and not embedded.is_global:
            raise JobInputError("Job URL must resolve only to public Internet addresses.")
        addresses.append(str(address))
    if not addresses:
        raise JobInputError("Could not resolve the public job hostname.")
    return parts, hostname, port, addresses[0]


def _fetch_public_page(url: str) -> RawJobPage:
    """Pin a validated IP on every hop, retaining TLS hostname verification.

    Direct pools deliberately bypass environment proxies, netrc credentials
    and cookies. Automatic redirects and retries are disabled.
    """
    deadline = time.monotonic() + FETCH_DEADLINE_SECONDS
    current_url = url
    visited = set()
    for _ in range(MAX_REDIRECTS + 1):
        if current_url in visited:
            raise JobInputError("The job page redirects in a loop.")
        visited.add(current_url)
        parts, hostname, port, address = _public_destination(current_url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise JobInputError("Job page retrieval exceeded its time limit.")
        timeout = urllib3.Timeout(total=remaining, connect=min(10, remaining), read=min(10, remaining))
        options = {"host": address, "port": port, "timeout": timeout, "retries": False}
        if parts.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(
                **options, server_hostname=hostname, assert_hostname=hostname, cert_reqs="CERT_REQUIRED",
            )
        else:
            pool = urllib3.HTTPConnectionPool(**options)
        host_header = f"[{hostname}]" if ":" in hostname else hostname
        if parts.port is not None:
            host_header += f":{port}"
        target = quote(urlunsplit(("", "", parts.path or "/", parts.query, "")), safe="/%?=&:+,;@!$'()*[]~-._")
        response = None
        try:
            response = pool.urlopen(
                "GET", target,
                headers={
                    "Host": host_header,
                    "User-Agent": "JobApplicationAgent/1.0",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Encoding": "identity",
                },
                redirect=False, retries=False, preload_content=False,
                assert_same_host=False,
            )
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise JobInputError("Job page redirect has no destination.")
                current_url = urljoin(current_url, location)
                continue
            if response.status != 200:
                raise JobInputError(f"Job page returned HTTP {response.status}.")
            content_type = response.headers.get("Content-Type", "").casefold()
            if content_type and not any(kind in content_type for kind in ("text/html", "application/xhtml+xml")):
                raise JobInputError("Job URL did not return an HTML page.")
            encoding = response.headers.get("Content-Encoding", "identity").casefold()
            if encoding not in {"", "identity"}:
                raise JobInputError("Job page ignored the uncompressed response request.")
            content_length = response.headers.get("Content-Length", "")
            if content_length.isdigit() and int(content_length) > MAX_PAGE_BYTES:
                raise JobInputError("Job page exceeds the 2 MB download limit.")
            chunks = []
            size = 0
            while True:
                if time.monotonic() >= deadline:
                    raise JobInputError("Job page retrieval exceeded its time limit.")
                chunk = response.read1(min(64_000, MAX_PAGE_BYTES + 1 - size), decode_content=False)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_PAGE_BYTES:
                    raise JobInputError("Job page exceeds the 2 MB download limit.")
                chunks.append(chunk)
            charset = re.search(r"charset=[\"']?([a-zA-Z0-9_.-]+)", content_type)
            try:
                html = b"".join(chunks).decode(charset.group(1) if charset else "utf-8", errors="replace")
            except LookupError:
                html = b"".join(chunks).decode("utf-8", errors="replace")
            return RawJobPage(url=current_url, html=html, source="application_url")
        except (urllib3.exceptions.HTTPError, OSError):
            raise JobInputError("Could not retrieve the public job page. Supply a full description file if access is blocked.") from None
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise JobInputError("Job page exceeded the redirect limit.")


class _ApplicationJsonLdSource(GenericJsonLdSource):
    def _to_job(self, job_posting: dict, page: RawJobPage):
        # JobPosting uses title; some employer pages supply schema.org name.
        normalized = dict(job_posting)
        normalized["title"] = job_posting.get("title") or job_posting.get("name")
        job = super()._to_job(normalized, page)
        job.metadata = {"extraction_method": "jobposting_jsonld"}
        return job


def _plausible_description(description: str) -> bool:
    return (
        len(description.strip()) >= MIN_DESCRIPTION_CHARS
        and len(description.split()) >= 60
        and not description.rstrip().endswith(("...", "…"))
    )


def fetch_job_url(url: str) -> dict:
    """Extract one public job URL without submitting forms or executing scripts.

    Complete-looking descriptions are marked full using conservative length
    and extraction checks. Short/aggregator results remain summary for REVIEW.
    """
    page = _fetch_public_page(url)
    source = DwpSource() if urlsplit(page.url).hostname == "findajob.dwp.gov.uk" else _ApplicationJsonLdSource()
    try:
        job = JobCleaner().clean(source.extract(page))
    except (AttributeError, TypeError, ValueError):
        raise JobInputError("The page has malformed job data; supply a full description file.") from None
    if not isinstance(job.title, str) or not job.title.strip() or not isinstance(job.description, str) or not job.description.strip():
        raise JobInputError("No job title and description were found. The page may require a browser/login.")
    metadata = dict(job.metadata or {})
    hostname = urlsplit(page.url).hostname or ""
    aggregator = "adzuna" in hostname.split(".")
    complete = _plausible_description(job.description) and not aggregator
    if job.source == "generic_html":
        # A long navigation/cookie page alone is not a job description.
        complete = complete and bool(re.search(
            r"\b(requirements?|responsibilit(?:y|ies)|qualifications?|experience|skills)\b",
            job.description, flags=re.IGNORECASE,
        ))
    metadata.update({
        "description_type": "full" if complete else "summary",
        "requested_url": url,
        "full_description_url": page.url if complete else None,
    })
    if not complete:
        metadata["enrichment_error"] = "The retrieved page does not contain a sufficiently complete employer description. Supply --description-file."
    job.metadata = metadata
    result = asdict(job)
    result["id"] = JobDeduplicator.url_fingerprint(page.url)[:16]
    result["canonical_url"] = JobDeduplicator.canonicalize_url(page.url)
    result["content_hash"] = JobDeduplicator.content_fingerprint(job)
    return result


def _words(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _same_role(original: dict, enriched: dict) -> bool:
    title_words = _words(original["title"])
    if len(title_words & _words(enriched.get("title") or "")) < max(1, len(title_words) / 2):
        return False
    if original.get("company") and enriched.get("company"):
        company_words = _words(str(original["company"])) - {"limited", "ltd", "inc", "plc", "corporation"}
        other_words = _words(str(enriched["company"]))
        if company_words and not company_words & other_words:
            return False
    return True


def _read_description(path: Path) -> str:
    path = Path(path)
    if path.suffix.lower() not in {".txt", ".md"}:
        raise JobInputError("Full description files must be .txt or .md.")
    try:
        if path.stat().st_size > MAX_DESCRIPTION_BYTES:
            raise JobInputError("Full description file exceeds the 250 KB limit.")
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise JobInputError("Cannot read the full description file as UTF-8 text.") from None
    return content


def prepare_job(
    job: dict,
    *,
    full_description_file: Path | None = None,
    fetch_full: bool = True,
) -> dict:
    """Hydrate a summary without changing its identity or the caller's record.

    ``metadata.original_job`` preserves the complete pre-enrichment record.
    Unresolved summaries retain their description and gain enrichment_error,
    allowing the application workflow to return REVIEW before any model call.
    """
    result = _validate_job(job)
    metadata = result["metadata"] = dict(result.get("metadata") or {})
    summary = metadata.get("description_type") == "summary" or result.get("source") == "adzuna" and metadata.get("description_type") != "full"
    if full_description_file is None and not summary:
        return result
    metadata["description_type"] = "summary"
    try:
        if full_description_file is not None:
            raw_description = _read_description(full_description_file)
            description = JobCleaner.clean_description(raw_description)
            full_url = result["source_url"]
        elif fetch_full:
            enriched = fetch_job_url(result["source_url"])
            if enriched.get("metadata", {}).get("description_type") != "full":
                raise JobInputError("The retrieved page still contains a summary. Supply --description-file.")
            if not _same_role(result, enriched):
                raise JobInputError("The retrieved page appears to describe a different role or employer. Supply --description-file.")
            description = enriched["description"]
            raw_description = enriched.get("raw_description") or description
            full_url = enriched["source_url"]
        else:
            raise JobInputError("Only a summary is available; enable full-page fetching or supply --description-file.")
        if not _plausible_description(description):
            raise JobInputError("The supplied description is too short or appears truncated. Supply a complete --description-file.")
        previous = " ".join(result["description"].split())
        normalized = " ".join(description.split())
        if summary and (normalized == previous or len(normalized) < len(previous) + 100):
            raise JobInputError("The fetched description does not add enough detail beyond the original summary. Supply --description-file.")
        original = deepcopy(job)
        metadata.setdefault("original_job", original)
        metadata.update({"description_type": "full", "full_description_url": full_url})
        metadata.pop("enrichment_error", None)
        if full_description_file is not None:
            metadata["description_file"] = str(full_description_file)
        result["description"] = description
        result["raw_description"] = raw_description
    except JobInputError as error:
        metadata["enrichment_error"] = str(error)
    return result
