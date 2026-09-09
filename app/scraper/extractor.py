from __future__ import annotations

from bs4 import BeautifulSoup

from app.scraper.sources.base import ExtractedJob, RawJobPage


class HtmlJobExtractor:
    def extract(self, page: RawJobPage) -> ExtractedJob:
        soup = BeautifulSoup(page.html, "html.parser")

        title = self._extract_title(soup)
        description = self._extract_description(
            soup,
            title,
        )

        return ExtractedJob(
            title=title,
            company=None,
            location=self._extract_location(soup),
            description=description,
            date_posted=None,
            salary=self._extract_salary(soup),
            source="generic_html",
            source_url=page.url,
        )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str | None:
        h1 = soup.find("h1")

        if h1:
            return h1.get_text(" ", strip=True)

        return None

    @staticmethod
    def _extract_location(soup: BeautifulSoup) -> str | None:
        text = soup.get_text("\n", strip=True)

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for index, line in enumerate(lines):
            if line.lower() == "location":
                if index + 1 < len(lines):
                    return lines[index + 1]

        return None

    @staticmethod
    def _extract_salary(soup: BeautifulSoup) -> str | None:
        text = soup.get_text("\n", strip=True)

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for index, line in enumerate(lines):
            if line.lower() == "salary":
                if index + 1 < len(lines):
                    return lines[index + 1]

        return None

    @staticmethod
    def _extract_description(
        soup: BeautifulSoup,
        title: str | None,
    ) -> str | None:

        if not title:
            return None

        matching_headings = []

        for heading in soup.find_all(
            ["h1", "h2", "h3", "h4"]
        ):
            heading_text = heading.get_text(
                " ",
                strip=True,
            )

            if heading_text == title:
                matching_headings.append(heading)

        if not matching_headings:
            return None

        # Prefer the last title occurrence.
        start_heading = matching_headings[-1]

        collected = []

        stop_markers = {
            "related jobs",
            "similar jobs",
            "other jobs",
            "recommended jobs",
        }

        for element in start_heading.find_all_next():

            if element.name in {"h2", "h3"}:
                heading_text = element.get_text(
                    " ",
                    strip=True,
                ).lower()

                if heading_text in stop_markers:
                    break

            if element.name in {
                "p",
                "li",
                "h3",
                "h4",
            }:
                text = element.get_text(
                    " ",
                    strip=True,
                )

                if text:
                    collected.append(text)

        return "\n".join(collected) or None