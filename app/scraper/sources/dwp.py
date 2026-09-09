from __future__ import annotations

from urllib.parse import urlencode, urljoin

from bs4 import BeautifulSoup

from app.scraper.crawler import JobCrawler
from app.scraper.sources.base import JobSource, RawJobPage, ExtractedJob


class DwpSource(JobSource):
    name = "dwp"

    BASE_URL = "https://findajob.dwp.gov.uk"
    SEARCH_URL = f"{BASE_URL}/search"

    def __init__(self, crawler: JobCrawler | None = None):
        self.crawler = crawler or JobCrawler()

    def fetch(self, url: str) -> RawJobPage:
        return self.crawler.fetch(
            url=url,
            source=self.name,
        )

    def discover_jobs(
    self,
    query: str,
    location: str = "",
    limit: int = 50,
) -> list[str]:

        discovered: list[str] = []
        seen: set[str] = set()

        page_number = 1

        while len(discovered) < limit:

            params = {
                "q": query,
                "w": location,
            }

            # DWP pagination starts being necessary
            # after the first page.
            if page_number > 1:
                params["p"] = page_number

            search_url = (
                f"{self.SEARCH_URL}?"
                f"{urlencode(params)}"
            )

            print(
                f"Fetching search page {page_number}: "
                f"{search_url}"
            )

            page = self.fetch(search_url)

            urls = self._extract_job_urls(
                page.html
            )

            print(
                f"Found {len(urls)} job links "
                f"on page {page_number}"
            )

            if not urls:
                break

            new_urls = 0

            for url in urls:

                if url in seen:
                    continue

                seen.add(url)
                discovered.append(url)
                new_urls += 1

                if len(discovered) >= limit:
                    break

            if new_urls == 0:
                break

            page_number += 1

        return discovered

    def _extract_job_urls(self, html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")

        urls: list[str] = []

        for link in soup.find_all("a", href=True):
            href = link["href"]

            if not isinstance(href, str):
                continue

            # DWP individual vacancies normally use /details/<id>
            if "/details/" not in href:
                continue

            absolute_url = urljoin(
                self.BASE_URL,
                href,
            )

            urls.append(absolute_url)

        return urls

    def extract(self, page: RawJobPage) -> ExtractedJob:
        """
        Temporary implementation.

        We will improve DWP-specific extraction after we inspect
        a real vacancy page.
        """
        soup = BeautifulSoup(
            page.html,
            "html.parser",
        )

        title = None

        h1 = soup.find("h1")

        if h1:
            title = h1.get_text(
                " ",
                strip=True,
            )

        description = self._extract_description(soup)

        return ExtractedJob(
            title=title,
            company=self._extract_field(
                soup,
                "Company",
            ),
            location=self._extract_field(
                soup,
                "Location",
            ),
            description=description,
            date_posted=None,
            salary=self._extract_field(
                soup,
                "Salary",
            ),
            source=self.name,
            source_url=page.url,
        )

    @staticmethod
    def _extract_field(
        soup: BeautifulSoup,
        label: str,
    ) -> str | None:
        text = soup.get_text(
            "\n",
            strip=True,
        )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        label_lower = label.lower()

        for index, line in enumerate(lines):
            if line.lower() == label_lower:
                if index + 1 < len(lines):
                    return lines[index + 1]

        return None

    @staticmethod
    def _extract_description(
        soup: BeautifulSoup,
    ) -> str | None:

        # First attempt: find a likely main-content block.
        main = soup.find("main")

        if main:
            text = main.get_text(
                separator="\n",
                strip=True,
            )

            if len(text) > 300:
                return text

        return None