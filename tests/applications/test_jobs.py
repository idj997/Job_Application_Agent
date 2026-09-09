from copy import deepcopy
import io
import json
import socket

import pytest

from app.applications import jobs
from app.scraper.sources.base import RawJobPage


FULL_DESCRIPTION = (
    "We are recruiting a Data Engineer to improve data services across our business. "
    "Responsibilities include building Python data pipelines, writing well tested SQL, "
    "and working with analysts to deliver reliable datasets for customer reporting. "
    "You will investigate failures, improve monitoring, and document how systems operate. "
    "Requirements include practical Python experience, confidence with relational databases, "
    "and clear written communication. You should enjoy collaborating with engineers and "
    "sharing your knowledge through thoughtful code reviews. The team works in London "
    "with two office days each week. We provide a learning budget, flexible hours, "
    "and time to develop new skills. Applications should include your current CV."
)


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Offline tests must not access the network")
    monkeypatch.setattr(socket, "getaddrinfo", unexpected)


def summary_job():
    return {
        "id": "original-id", "source": "adzuna", "external_id": "original-external",
        "source_url": "https://www.adzuna.co.uk/land/ad/123",
        "canonical_url": "https://www.adzuna.co.uk/land/ad/123",
        "title": "Data Engineer", "company": "Example Ltd", "location": "London",
        "description": "Build Python data pipelines and write SQL.",
        "raw_description": "<p>Build Python data pipelines and write SQL.</p>",
        "metadata": {"description_type": "summary", "salary_min": 40000},
        "scraped_at": "2026-09-09T10:00:00Z",
    }


def full_job(**changes):
    result = summary_job()
    result.update({"source": "generic_jsonld", "source_url": "https://example.com/jobs/123", "description": FULL_DESCRIPTION})
    result["metadata"] = {"description_type": "full"}
    result.update(changes)
    return result


def test_load_existing_storage_shape_and_report_all_bad_files(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(summary_job()))
    assert jobs.load_jobs(tmp_path)[0] == summary_job()
    (tmp_path / "array.json").write_text("[]")
    (tmp_path / "broken.json").write_text("bad json")
    with pytest.raises(jobs.JobInputError) as error:
        jobs.load_jobs(tmp_path)
    assert "array.json" in str(error.value)
    assert "broken.json" in str(error.value)
    assert "good.json" not in str(error.value)


@pytest.mark.parametrize("changes", [{"title": None}, {"description": []}, {"metadata": []}, {"source_url": "file:///etc/passwd"}])
def test_invalid_job_shapes_are_rejected(changes):
    record = summary_job()
    record.update(changes)
    with pytest.raises(jobs.JobInputError):
        jobs.prepare_job(record, fetch_full=False)


def test_empty_job_directory_is_actionable(tmp_path):
    with pytest.raises(jobs.JobInputError, match="No job JSON"):
        jobs.load_jobs(tmp_path)


def test_explicit_job_list_takes_precedence(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(summary_job()))
    assert jobs.load_jobs(tmp_path / "missing", [path])[0]["id"] == "original-id"


def test_complete_jobs_are_unchanged_and_not_fetched(monkeypatch):
    record = full_job()
    before = deepcopy(record)
    assert jobs.prepare_job(record) == before
    assert record == before


def test_summary_enrichment_preserves_identity_and_complete_original(monkeypatch):
    monkeypatch.setattr(jobs, "fetch_job_url", lambda url: full_job())
    record = summary_job()
    before = deepcopy(record)

    enriched = jobs.prepare_job(record)

    assert record == before
    for field in ("id", "source", "external_id", "source_url", "canonical_url", "title", "company", "scraped_at"):
        assert enriched[field] == before[field]
    assert enriched["description"] == FULL_DESCRIPTION
    assert enriched["metadata"]["description_type"] == "full"
    assert enriched["metadata"]["salary_min"] == 40000
    assert enriched["metadata"]["full_description_url"] == "https://example.com/jobs/123"
    assert enriched["metadata"]["original_job"] == before


def test_summary_without_fetch_stays_summary_for_review():
    record = summary_job()
    result = jobs.prepare_job(record, fetch_full=False)
    assert result["metadata"]["description_type"] == "summary"
    assert "--description-file" in result["metadata"]["enrichment_error"]
    assert result["description"] == record["description"]


def test_network_failure_does_not_promote_summary(monkeypatch):
    def unavailable(url):
        raise jobs.JobInputError("Job page returned HTTP 403.")
    monkeypatch.setattr(jobs, "fetch_job_url", unavailable)
    result = jobs.prepare_job(summary_job())
    assert result["metadata"]["description_type"] == "summary"
    assert "403" in result["metadata"]["enrichment_error"]


def test_identical_long_summary_is_not_promoted(monkeypatch):
    record = summary_job()
    record["description"] = FULL_DESCRIPTION
    monkeypatch.setattr(jobs, "fetch_job_url", lambda url: full_job())
    result = jobs.prepare_job(record)
    assert result["metadata"]["description_type"] == "summary"
    assert "does not add enough detail" in result["metadata"]["enrichment_error"]


@pytest.mark.parametrize("changes", [{"title": "Marketing Director"}, {"company": "Unrelated Widgets"}])
def test_different_role_or_employer_is_not_promoted(monkeypatch, changes):
    monkeypatch.setattr(jobs, "fetch_job_url", lambda url: full_job(**changes))
    result = jobs.prepare_job(summary_job())
    assert result["metadata"]["description_type"] == "summary"
    assert "different role or employer" in result["metadata"]["enrichment_error"]


def test_explicit_description_file_hydrates_without_network(tmp_path):
    path = tmp_path / "description.txt"
    path.write_text(FULL_DESCRIPTION)
    record = summary_job()
    result = jobs.prepare_job(record, full_description_file=path)
    assert result["description"] == FULL_DESCRIPTION
    assert result["metadata"]["description_type"] == "full"
    assert result["metadata"]["original_job"] == record


def test_secret_or_unsupported_description_file_is_not_read(tmp_path):
    path = tmp_path / ".env"
    path.write_text("SECRET_MUST_NOT_BE_READ")
    result = jobs.prepare_job(summary_job(), full_description_file=path)
    assert result["metadata"]["description_type"] == "summary"
    assert ".txt or .md" in result["metadata"]["enrichment_error"]
    assert "SECRET_MUST_NOT_BE_READ" not in str(result)


def test_jsonld_name_fallback_and_raw_description_preserved(monkeypatch):
    payload = {"@type": "JobPosting", "name": "Data Engineer", "description": f"<p>{FULL_DESCRIPTION}</p>"}
    page = RawJobPage(url="https://employer.example/jobs/123", source="test", html=f'<script type="application/ld+json">{json.dumps(payload)}</script>')
    monkeypatch.setattr(jobs, "_fetch_public_page", lambda url: page)
    result = jobs.fetch_job_url("https://employer.example/redirect")
    assert result["title"] == "Data Engineer"
    assert result["description"] == FULL_DESCRIPTION
    assert result["raw_description"] == f"<p>{FULL_DESCRIPTION}</p>"
    assert result["metadata"]["description_type"] == "full"
    assert result["metadata"]["requested_url"] == "https://employer.example/redirect"
    assert result["metadata"]["extraction_method"] == "jobposting_jsonld"


@pytest.mark.parametrize("url,description", [
    ("https://employer.example/job", "A short summary"),
    ("https://www.adzuna.co.uk/job", FULL_DESCRIPTION),
    ("https://employer.example/job", FULL_DESCRIPTION + " ..."),
])
def test_short_truncated_or_aggregator_pages_stay_summary(monkeypatch, url, description):
    page = RawJobPage(url=url, source="test", html=f"<h1>Data Engineer</h1><p>{description}</p>")
    monkeypatch.setattr(jobs, "_fetch_public_page", lambda url: page)
    result = jobs.fetch_job_url(url)
    assert result["metadata"]["description_type"] == "summary"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/job", "http://user:password@example.com/job",
    "http://localhost/job", "http://server.local/job", "http://example.com:8080/job",
    "http://example.com/\njob", "http://example.com\\@localhost/job",
])
def test_unsafe_urls_rejected_before_any_network(url):
    with pytest.raises(jobs.JobInputError):
        jobs._fetch_public_page(url)


def resolved(address):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return (family, socket.SOCK_STREAM, 6, "", (address, 443))


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "169.254.169.254", "192.168.1.20", "::1", "fc00::1", "2002:7f00:1::1"])
def test_private_and_embedded_private_destinations_rejected(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [resolved(address)])
    with pytest.raises(jobs.JobInputError, match="public Internet"):
        jobs._fetch_public_page("https://example.com/job")


def test_mixed_public_and_private_dns_answers_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [resolved("93.184.216.34"), resolved("10.0.0.1")])
    with pytest.raises(jobs.JobInputError, match="public Internet"):
        jobs._fetch_public_page("https://example.com/job")


class FakeResponse:
    def __init__(self, status=200, body=b"<h1>Job</h1>", headers=None):
        self.status = status
        self.body = io.BytesIO(body)
        self.headers = {"Content-Type": "text/html"} if headers is None else headers
        self.closed = False

    def read1(self, amount, decode_content=False):
        return self.body.read(amount)

    def close(self):
        self.closed = True


def fake_network(monkeypatch, *responses):
    records = []
    replies = iter(responses)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [resolved("93.184.216.34")])

    class Pool:
        def __init__(self, **kwargs):
            self.record = {"options": kwargs, "closed": False}
            records.append(self.record)

        def urlopen(self, *args, **kwargs):
            self.record["request"] = (args, kwargs)
            return next(replies)

        def close(self):
            self.record["closed"] = True

    monkeypatch.setattr(jobs.urllib3, "HTTPSConnectionPool", Pool)
    monkeypatch.setattr(jobs.urllib3, "HTTPConnectionPool", Pool)
    return records


def test_fetch_pins_public_ip_and_preserves_tls_host_checks(monkeypatch):
    response = FakeResponse()
    records = fake_network(monkeypatch, response)
    page = jobs._fetch_public_page("https://employer.example/jobs/123?q=python")
    assert page.html == "<h1>Job</h1>"
    assert records[0]["options"]["host"] == "93.184.216.34"
    assert records[0]["options"]["server_hostname"] == "employer.example"
    assert records[0]["options"]["assert_hostname"] == "employer.example"
    assert records[0]["options"]["cert_reqs"] == "CERT_REQUIRED"
    args, kwargs = records[0]["request"]
    assert args == ("GET", "/jobs/123?q=python")
    assert kwargs["headers"]["Host"] == "employer.example"
    assert kwargs["redirect"] is False
    assert kwargs["retries"] is False
    assert kwargs["preload_content"] is False
    assert records[0]["closed"] and response.closed


def test_redirect_to_private_destination_is_never_requested(monkeypatch):
    redirect = FakeResponse(status=302, headers={"Location": "http://localhost/private"})
    records = fake_network(monkeypatch, redirect)
    with pytest.raises(jobs.JobInputError, match="public Internet"):
        jobs._fetch_public_page("https://employer.example/job")
    assert len(records) == 1
    assert redirect.closed


def test_relative_redirect_is_validated_and_records_final_url(monkeypatch):
    records = fake_network(monkeypatch, FakeResponse(status=302, headers={"Location": "/jobs/final"}), FakeResponse())
    page = jobs._fetch_public_page("https://employer.example/job")
    assert page.url == "https://employer.example/jobs/final"
    assert len(records) == 2


@pytest.mark.parametrize("headers,body,match", [
    ({"Content-Type": "application/json"}, b"{}", "HTML page"),
    ({"Content-Encoding": "gzip"}, b"gzip", "uncompressed"),
    ({"Content-Length": "3000000"}, b"", "2 MB"),
    ({}, b"x" * (jobs.MAX_PAGE_BYTES + 1), "2 MB"),
])
def test_download_type_encoding_and_size_limits(monkeypatch, headers, body, match):
    reply = FakeResponse(headers=headers, body=body)
    fake_network(monkeypatch, reply)
    with pytest.raises(jobs.JobInputError, match=match):
        jobs._fetch_public_page("https://employer.example/job")
    assert reply.closed


def test_redirect_loop_is_bounded(monkeypatch):
    records = fake_network(monkeypatch, FakeResponse(status=302, headers={"Location": "/job"}))
    with pytest.raises(jobs.JobInputError, match="loop"):
        jobs._fetch_public_page("https://employer.example/job")
    assert len(records) == 1
