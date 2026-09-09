class ScraperError(Exception):
    """Base exception for job-ingestion failures."""


class SourceUnavailableError(ScraperError):
    """Raised when an upstream source cannot be queried."""


class SourceConfigurationError(ScraperError):
    """Raised when a source is missing required local configuration."""


class ExtractionError(ScraperError):
    """Raised when a source payload cannot be normalized."""


class InvalidJobError(ScraperError):
    """Raised when a normalized job fails the basic quality gate."""
