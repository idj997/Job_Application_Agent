"""Durable application state and repeat-submission protection."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_PROTECTED_STATUSES = {"submitted", "submission_unknown"}
_TRACKING_PARAMETERS = {
    "gclid", "fbclid", "msclkid", "ref", "source", "gh_src", "lever-source",
}


def _normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def job_key(job: dict) -> str:
    """Identify a vacancy without treating tracking links as separate jobs."""
    metadata = job.get("metadata") or {}
    company = _normalized(job.get("company") or metadata.get("company_slug"))
    requisition = _normalized(job.get("requisition_id") or metadata.get("requisition_id"))
    if company and requisition:
        identity = ["requisition", company, requisition]
    else:
        url = str(job.get("application_url") or metadata.get("apply_url")
                  or job.get("canonical_url") or job.get("source_url") or "").strip()
        parts = urlsplit(url)
        if parts.scheme.casefold() not in {"http", "https"} or not parts.hostname:
            source = _normalized(job.get("source"))
            external_id = _normalized(job.get("external_id"))
            if not (source and external_id):
                raise ValueError("A job needs a valid source URL or a source and external ID")
            identity = ["source", source, company, external_id]
        else:
            query = sorted(
                (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
                if not key.casefold().startswith("utm_")
                and key.casefold() not in _TRACKING_PARAMETERS
            )
            # Lever links may point at either the posting or its /apply form.
            path = parts.path.rstrip("/")
            if parts.hostname.casefold() in {"jobs.lever.co", "jobs.eu.lever.co"}:
                path = path.removesuffix("/apply")
            canonical = urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(),
                                   path, urlencode(query), ""))
            identity = ["url", canonical]
    encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ApplicationLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS applications ("
                "job_key TEXT PRIMARY KEY, status TEXT NOT NULL, "
                "record_json TEXT NOT NULL, created_at TEXT NOT NULL, "
                "updated_at TEXT NOT NULL)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, job_key: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM applications WHERE job_key = ?", (job_key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM applications ORDER BY updated_at DESC, job_key"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def claim_open(self, job_key: str) -> dict:
        """Atomically claim one ready application and retain a durable attempt marker.

        Return the original ready record for the browser. A second process cannot
        claim the same application, even if it read a ready record earlier.
        """
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_json FROM applications WHERE job_key = ?", (job_key,)
            ).fetchone()
            if not row:
                raise KeyError(job_key)
            original = json.loads(row[0])
            if original["status"] != "ready" or original.get("submission_attempted"):
                raise ValueError("Application is already opened, attempted, or not ready")
            now = datetime.now(timezone.utc).isoformat()
            claimed = {**original, "status": "opened", "submission_attempted": True,
                       "opened_at": now, "updated_at": now}
            connection.execute(
                "UPDATE applications SET status=?, record_json=?, updated_at=? WHERE job_key=?",
                ("opened", json.dumps(claimed, ensure_ascii=False, allow_nan=False), now, job_key),
            )
            return original

    def save(self, job_key: str, record: dict) -> None:
        """Upsert preparation state; preserve confirmed or uncertain submissions."""
        if not job_key:
            raise ValueError("job_key cannot be empty")
        if record.get("status") == "submitted":
            raise ValueError("Use mark_submitted with a receipt to record a submission")
        if not isinstance(record.get("status"), str) or not record["status"].strip():
            raise ValueError("An application record needs a status")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT status, created_at, record_json FROM applications WHERE job_key = ?", (job_key,)
            ).fetchone()
            previous_record = json.loads(previous[2]) if previous else {}
            if previous and (previous[0] in _PROTECTED_STATUSES
                             or previous_record.get("submission_attempted")):
                return
            now = datetime.now(timezone.utc).isoformat()
            payload = dict(record)
            if previous_record.get("resolution_history"):
                payload["resolution_history"] = previous_record["resolution_history"]
            payload.update(job_key=job_key, created_at=previous[1] if previous else now,
                           updated_at=now)
            connection.execute(
                "INSERT INTO applications (job_key, status, record_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(job_key) DO UPDATE SET "
                "status=excluded.status, record_json=excluded.record_json, "
                "updated_at=excluded.updated_at",
                (job_key, payload["status"], json.dumps(payload, ensure_ascii=False, allow_nan=False),
                 payload["created_at"], now),
            )

    def mark_submitted(self, job_key: str, confirmation: str) -> None:
        if not isinstance(confirmation, str) or not confirmation.strip():
            raise ValueError("A nonempty submission confirmation is required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_json FROM applications WHERE job_key = ?", (job_key,)
            ).fetchone()
            if not row:
                raise KeyError(job_key)
            record = json.loads(row[0])
            if record["status"] == "submitted":
                return
            if record["status"] not in {"ready", "opened", "submission_unknown"}:
                raise ValueError("Only ready, opened, or uncertain applications can be marked submitted")
            now = datetime.now(timezone.utc).isoformat()
            record.update(status="submitted", confirmation=confirmation.strip(),
                          submitted_at=now, updated_at=now)
            connection.execute(
                "UPDATE applications SET status=?, record_json=?, updated_at=? WHERE job_key=?",
                ("submitted", json.dumps(record, ensure_ascii=False, allow_nan=False), now, job_key),
            )

    def resolve_not_submitted(self, job_key: str, note: str) -> None:
        """Allow retry only after the applicant explicitly checked the employer outcome."""
        if not isinstance(note, str) or not note.strip():
            raise ValueError("A nonempty note explaining the checked outcome is required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_json FROM applications WHERE job_key = ?", (job_key,)
            ).fetchone()
            if not row:
                raise KeyError(job_key)
            record = json.loads(row[0])
            if record["status"] == "submitted":
                raise ValueError("A confirmed submission cannot be reset")
            if record["status"] not in {"opened", "submission_unknown"}:
                raise ValueError("Only opened or uncertain applications need outcome resolution")
            now = datetime.now(timezone.utc).isoformat()
            history = list(record.get("resolution_history") or [])
            history.append({"resolved_at": now, "previous_status": record["status"],
                            "outcome": "not_submitted", "note": note.strip()})
            record.update(status="ready", updated_at=now, resolution_history=history)
            record.pop("submission_attempted", None)
            connection.execute(
                "UPDATE applications SET status=?, record_json=?, updated_at=? WHERE job_key=?",
                ("ready", json.dumps(record, ensure_ascii=False, allow_nan=False), now, job_key),
            )
