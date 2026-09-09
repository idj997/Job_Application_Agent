from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.scraper.sources.base import RawJobPage


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}


class JobCrawler:
    def __init__(
        self,
        timeout: int = 20,
        headers: dict[str, str] | None = None,
    ):
        self.timeout = timeout
        self.headers = headers or DEFAULT_HEADERS

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        retry = Retry(
            total=4,
            connect=3,
            read=3,
            status=4,
            backoff_factor=2,
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(
            max_retries=retry,
        )

        self.session.mount(
            "https://",
            adapter,
        )

        self.session.mount(
            "http://",
            adapter,
        )

    def fetch(
        self,
        url: str,
        source: str = "generic",
    ) -> RawJobPage:

        response = self.session.get(
            url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return RawJobPage(
            url=response.url,
            html=response.text,
            source=source,
            status_code=response.status_code,
        )