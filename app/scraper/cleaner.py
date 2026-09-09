from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

from app.scraper.sources.base import ExtractedJob


class JobCleaner:
    def clean(self, job: ExtractedJob) -> ExtractedJob:
        if job.title:
            job.title = self.clean_text(job.title)

        if job.company:
            job.company = self.clean_text(job.company)

        if job.location:
            job.location = self.clean_text(job.location)

        if job.description:
            if job.raw_description is None:
                job.raw_description = job.description
            job.description = self.clean_description(job.description)

        return job

    @staticmethod
    def clean_html(value: str) -> str:
        return JobCleaner.clean_description(value)

    @staticmethod
    def clean_text(value: str) -> str:
        value = html.unescape(value)

        return JobCleaner.normalise_whitespace(value)

    @staticmethod
    def normalise_whitespace(value: str) -> str:
        value = value.replace("\r\n", "\n")
        value = value.replace("\r", "\n")

        value = re.sub(
            r"[ \t]+",
            " ",
            value,
        )

        value = re.sub(
            r"\n{3,}",
            "\n\n",
            value,
        )

        return value.strip()

    @staticmethod
    def clean_description(value: str) -> str:
        """
        Minimal cleaning only.

        Preserve realistic job-description noise because
        downstream classifiers should be robust to it.
        """

        value = html.unescape(value)

        soup = BeautifulSoup(value, "html.parser")

        # Inline formatting is not a text boundary. Unwrapping it before
        # extraction prevents phrases such as ``strong SQL experience``
        # from being split across several lines.
        for tag in soup.find_all(
            ["a", "abbr", "b", "code", "em", "i", "mark", "small", "span", "strong"]
        ):
            tag.unwrap()
        soup.smooth()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        # Normalize excessive spaces.
        text = re.sub(r"[ \t]+", " ", text)

        # Don't allow ridiculous numbers of blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def is_valid_job(job: ExtractedJob) -> bool:
        if not job.title:
            return False

        if not job.description:
            return False

        if len(job.description) < 300:
            return False

        return True
