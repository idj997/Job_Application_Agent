from app.scraper.cleaner import JobCleaner
from app.scraper.sources.base import RawJobPage
from app.scraper.sources.generic_jsonld import GenericJsonLdSource


def test_generic_jsonld_extraction_still_preserves_raw_description():
    page = RawJobPage(
        url="https://example.com/jobs/123?utm_source=test",
        source="test",
        html="""
            <html><head>
            <script type="application/ld+json">
            {
              "@type": "JobPosting",
              "title": "Data Engineer",
              "description": "<p>Build reliable <strong>data pipelines</strong>.</p>",
              "datePosted": "2026-08-18",
              "hiringOrganization": {"name": "Example Ltd"},
              "jobLocation": {
                "address": {
                  "addressLocality": "London",
                  "addressCountry": "United Kingdom"
                }
              }
            }
            </script>
            </head></html>
        """,
    )

    job = GenericJsonLdSource().extract(page)
    cleaned = JobCleaner().clean(job)

    assert cleaned.title == "Data Engineer"
    assert cleaned.company == "Example Ltd"
    assert cleaned.location == "London, United Kingdom"
    assert cleaned.raw_description == (
        "<p>Build reliable <strong>data pipelines</strong>.</p>"
    )
    assert cleaned.description == "Build reliable data pipelines."
