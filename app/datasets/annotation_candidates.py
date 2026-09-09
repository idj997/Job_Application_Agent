from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.datasets.requirement_dataset import (
    DatasetValidationError,
    LABELS,
    RequirementExample,
)
from app.scraper.role_catalog import RoleCatalog, RoleCatalogError


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("Python",),
    "SQL": ("SQL",),
    "Java": ("Java",),
    "Scala": ("Scala",),
    "C++": ("C++",),
    "C": ("C language", "programming in C"),
    "Rust": ("Rust",),
    "Go": ("Golang", "Go language"),
    "R": ("R language", "programming in R"),
    "MATLAB": ("MATLAB",),
    "PySpark": ("PySpark",),
    "Apache Spark": ("Apache Spark", "Spark"),
    "Apache Kafka": ("Apache Kafka", "Kafka"),
    "Apache Airflow": ("Apache Airflow", "Airflow"),
    "dbt": ("dbt",),
    "Snowflake": ("Snowflake",),
    "Databricks": ("Azure Databricks", "Databricks"),
    "Azure Data Factory": ("Azure Data Factory",),
    "ETL": ("ETL", "extract transform load"),
    "ELT": ("ELT", "extract load transform"),
    "data modelling": ("data modelling", "data modeling"),
    "data warehousing": ("data warehousing", "data warehouse"),
    "machine learning": ("machine learning",),
    "deep learning": ("deep learning",),
    "PyTorch": ("PyTorch",),
    "TensorFlow": ("TensorFlow",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "MLOps": ("MLOps", "ML Ops"),
    "MLflow": ("MLflow",),
    "natural language processing": ("natural language processing", "NLP"),
    "computer vision": ("computer vision",),
    "generative AI": ("generative AI", "GenAI"),
    "large language models": ("large language models", "LLMs", "LLM"),
    "retrieval-augmented generation": ("retrieval-augmented generation", "RAG"),
    "prompt engineering": ("prompt engineering",),
    "CUDA": ("CUDA",),
    "ROCm": ("ROCm",),
    "GPU architecture": ("GPU architecture", "GPU architectures"),
    "CPU architecture": ("CPU architecture", "CPU architectures"),
    "microarchitecture": ("microarchitecture",),
    "RTL": ("RTL", "register-transfer level"),
    "Verilog": ("Verilog",),
    "SystemVerilog": ("SystemVerilog",),
    "UVM": ("UVM", "Universal Verification Methodology"),
    "VHDL": ("VHDL",),
    "FPGA": ("FPGA", "FPGAs"),
    "ASIC": ("ASIC", "ASICs"),
    "RISC-V": ("RISC-V", "RISC V"),
    "ARM": ("ARM architecture", "ARM processors"),
    "x86": ("x86",),
    "cache coherence": ("cache coherence",),
    "compiler design": ("compiler design", "compiler development"),
    "AWS": ("Amazon Web Services", "AWS"),
    "Microsoft Azure": ("Microsoft Azure", "Azure"),
    "Google Cloud": ("Google Cloud Platform", "Google Cloud", "GCP"),
    "Docker": ("Docker",),
    "Kubernetes": ("Kubernetes",),
    "Terraform": ("Terraform",),
    "Linux": ("Linux",),
    "Git": ("Git",),
    "CI/CD": ("CI/CD", "continuous integration", "continuous delivery"),
    "REST APIs": ("REST APIs", "RESTful APIs", "REST API"),
    "distributed systems": ("distributed systems",),
    "software engineering": (
        "software engineering",
        "software development",
        "software programming",
    ),
    "backend engineering": (
        "backend engineering",
        "backend development",
        "backend systems",
    ),
    "cloud engineering": ("cloud engineering", "cloud infrastructure"),
    "cybersecurity": ("cybersecurity", "cyber security"),
    "security engineering": ("security engineering",),
    "cloud security": ("cloud security",),
    "network security": ("network security", "network segmentation"),
    "application security": ("application security",),
    "firewalling": ("firewalling", "firewalls"),
    "threat modelling": ("threat modelling", "threat modeling"),
    "penetration testing": ("penetration testing",),
    "incident response": ("incident response",),
    "GPU computing": (
        "GPU computing",
        "GPU systems",
        "GPU programming",
        "GPU engineering",
        "GPU engineer",
    ),
    "GPU compiler": ("GPU compiler",),
    "processor design": ("processor design", "CPU design"),
    "CPU verification": ("CPU verification", "processor verification"),
    "performance modelling": ("performance modelling", "performance modeling"),
    "experimental design": (
        "experimental design",
        "experimental methods",
        "bandit algorithms",
    ),
    "authentication and authorization": (
        "authentication/authorization",
        "authentication and authorization",
        "authentication or authorization",
    ),
    "enterprise systems": ("enterprise systems",),
}


LABEL_RULES: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    (
        "NOT_REQUIREMENT",
        0.98,
        (
            r"\bnot\s+(?:strictly\s+)?required\b",
            r"\bno\s+(?:previous|prior)?\s*.{0,45}\b(?:required|needed|necessary)\b",
            r"\bdo(?:es)?\s+not\s+(?:need|require)\b",
            r"\bwithout\s+(?:previous|prior)?\s*.{0,35}\bexperience\b",
        ),
    ),
    (
        "PREFERRED",
        0.94,
        (
            r"\bpreferred\b",
            r"\bdesirable\b",
            r"\badvantageous\b",
            r"\bnice[ -]to[ -]have\b",
            r"\bwould be (?:useful|beneficial)\b",
            r"\b(?:is|are|a) (?:big )?plus\b",
            r"\bbonus\b",
        ),
    ),
    (
        "CORE",
        0.96,
        (
            r"\brequired\b",
            r"\bessential\b",
            r"\bmandatory\b",
            r"\bmust(?:\s+have|\s+be|\s+demonstrate|\s+possess)?\b",
            r"\bprerequisite\b",
            r"\bwe require\b",
            r"\brequirements?\s*:",
        ),
    ),
    (
        "IMPORTANT",
        0.86,
        (
            r"\byou(?:'ll| will)\b",
            r"\bresponsible for\b",
            r"\bexpected to\b",
            r"\bstrong (?:experience|knowledge|proficiency|skills?)\b",
            r"\bproven (?:experience|ability|track record)\b",
            r"\badvanced (?:experience|knowledge|proficiency|skills?)\b",
            r"\bhands-on experience\b",
        ),
    ),
    (
        "IMPORTANT",
        0.78,
        (
            r"\bexperience (?:with|in|using)\b",
            r"\bknowledge of\b",
            r"\bexpertise (?:with|in)\b",
            r"\bproficien(?:t|cy) (?:with|in)\b",
        ),
    ),
    (
        "CONTEXTUAL",
        0.82,
        (
            r"\b(?:our|the) (?:team|platform team|platform|stack|environment) (?:uses?|includes?|runs?)\b",
            r"\b(?:is|are|was|were) built (?:using|with|on)\b",
            r"\bwork(?:ing)? alongside\b",
            r"\bfamiliarity with\b",
            r"\bawareness of\b",
            r"\bexposure to\b",
            r"\b(?:organization|organisation|company|business) "
            r"(?:builds?|operates?|develops?|uses?)\b",
        ),
    ),
    (
        "IMPORTANT",
        0.70,
        (
            r"\b(?:analyze|analyse|build|develop|design|implement|manage|deploy|maintain|review|"
            r"improve|secure|optimise|optimize)(?:s|ed|ing)?\b",
        ),
    ),
)


SECTION_LABEL_PATTERNS: tuple[
    tuple[str, float, tuple[re.Pattern[str], ...]], ...
] = (
    (
        "PREFERRED",
        0.80,
        (
            re.compile(
                r"^(?:preferred|desirable|nice[ -]to[ -]have)(?:\s+"
                r"(?:qualifications?|experience|skills?|attributes?))?\s*:?$",
                re.IGNORECASE,
            ),
            re.compile(
                r"^it is advantageous for you to have knowledge of the following\s*:?$",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "CORE",
        0.88,
        (
            re.compile(
                r"^(?:required|minimum|essential|mandatory)(?:\s+"
                r"(?:qualifications?|experience|skills?|requirements?))?\s*:?$",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "IMPORTANT",
        0.76,
        (
            re.compile(
                r"^(?:qualifications?|your qualifications|about you|who you are|"
                r"what you bring)(?:\s+to the role)?\s*:?$",
                re.IGNORECASE,
            ),
        ),
    ),
)

SECTION_BREAK_PATTERN = re.compile(
    r"^(?:about (?:the role|us)|responsibilities|what you(?:'ll| will) do|"
    r"the role|benefits|compensation|what we offer|day[ -]to[ -]day|"
    r"how (?:you|we) work)\s*:?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidateGenerationSummary:
    jobs_read: int
    jobs_failed: int
    candidates: int
    label_counts: dict[str, int]
    output_path: Path


@dataclass(frozen=True)
class CandidatePromotionSummary:
    candidates_read: int
    approved: int
    rejected: int
    pending: int
    output_path: Path
    decision_log_path: Path | None = None


@dataclass(frozen=True)
class CandidateReviewValidationSummary:
    candidates_read: int
    errors: int
    warnings: int
    information: int
    output_path: Path


ANNOTATION_STATUS_ALIASES = {
    "approve": "approved",
    "approved": "approved",
    "reject": "rejected",
    "rejected": "rejected",
    "needs_review": "needs_review",
    "needs review": "needs_review",
    "pending": "needs_review",
}

_NOTE_LABEL_PATTERN = re.compile(
    r"\b(?:core|important|preferred|contextual|not[ _-]?requirement)\b",
    re.IGNORECASE,
)


def normalize_annotation_status(value: Any) -> str:
    """Return a canonical review status without silently accepting typos."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return "needs_review"
    if not isinstance(value, str):
        raise DatasetValidationError("annotation_status must be a string or null")
    normalized = re.sub(r"\s+", " ", value.strip().casefold())
    try:
        return ANNOTATION_STATUS_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(ANNOTATION_STATUS_ALIASES))
        raise DatasetValidationError(
            f"unknown annotation_status {value!r}; expected one of {allowed}"
        ) from exc


def _normalize_label(value: Any, field_name: str) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise DatasetValidationError(f"{field_name} must be a string or null")
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized not in LABELS:
        allowed = ", ".join(LABELS)
        raise DatasetValidationError(
            f"{field_name} must be one of {allowed}; received {value!r}"
        )
    return normalized


def _labels_mentioned_in_notes(notes: str) -> list[str]:
    labels = {
        match.group(0).upper().replace("-", "_").replace(" ", "_")
        for match in _NOTE_LABEL_PATTERN.finditer(notes)
    }
    return sorted(label for label in labels if label in LABELS)


def read_candidate_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise DatasetValidationError(f"candidate file does not exist: {source}")
    content = source.read_text(encoding="utf-8")
    if not content.strip():
        return []

    if content.lstrip().startswith("["):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"invalid JSON array in {source}: {exc.msg}"
            ) from exc
        if not isinstance(payload, list) or not all(
            isinstance(record, dict) for record in payload
        ):
            raise DatasetValidationError(
                f"candidate JSON in {source} must be an array of objects"
            )
        return payload

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"invalid JSON in {source}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise DatasetValidationError(
                f"candidate at {source}:{line_number} must be an object"
            )
        records.append(record)
    return records


def write_candidate_records(
    records: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(records)
    if destination.suffix.casefold() == ".json":
        content = json.dumps(
            ordered,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    else:
        content = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in ordered
        )
    destination.write_text(content, encoding="utf-8")
    return destination


def _review_context_lines(
    description: str,
    evidence: str,
    surrounding_lines: int = 3,
) -> list[str]:
    lines = [line.strip() for line in description.splitlines()]
    evidence_line = next(
        (index for index, line in enumerate(lines) if evidence in line),
        None,
    )
    if evidence_line is None:
        return [evidence]

    start = max(0, evidence_line - surrounding_lines)
    end = min(len(lines), evidence_line + surrounding_lines + 1)
    context = [line for line in lines[start:end] if line]
    if start > 0:
        context.insert(0, "…")
    if end < len(lines):
        context.append("…")
    return context


def _compact_review_record(record: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "annotation_status": record.get("annotation_status"),
        "reviewed_label": record.get("reviewed_label"),
        "review_notes": record.get("review_notes"),
        "candidate_id": record.get("candidate_id"),
        "job_id": record.get("job_id"),
        "job_title": record.get("job_title"),
        "role_family": record.get("role_family"),
        "requirement": record.get("requirement"),
        "proposed_label": record.get("proposed_label"),
        "evidence_span": record.get("evidence_span"),
        "review_context": _review_context_lines(
            str(record.get("job_description") or ""),
            str(record.get("evidence_span") or ""),
        ),
    }
    excluded = set(compact) | {"job_description"}
    if not record.get("job_description_ref") and record.get("job_description"):
        compact["job_description"] = record["job_description"]
    compact.update(
        (key, value)
        for key, value in record.items()
        if key not in excluded
    )
    return compact


def _hydrate_job_description(
    candidate: dict[str, Any],
    candidate_path: Path,
    record_number: int,
) -> dict[str, Any]:
    description = candidate.get("job_description")
    if isinstance(description, str) and description.strip():
        return candidate

    reference = candidate.get("job_description_ref")
    if not isinstance(reference, str) or not reference.strip():
        raise DatasetValidationError(
            f"approved candidate at {candidate_path} record {record_number} "
            "requires job_description or job_description_ref"
        )
    job_path = Path(reference)
    try:
        job_payload = json.loads(job_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetValidationError(
            f"could not read job_description_ref {job_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"job_description_ref contains invalid JSON: {job_path}"
        ) from exc
    if not isinstance(job_payload, dict):
        raise DatasetValidationError(
            f"job_description_ref must contain a JSON object: {job_path}"
        )
    if job_payload.get("id") != candidate.get("job_id"):
        raise DatasetValidationError(
            f"job_description_ref ID does not match candidate at "
            f"{candidate_path} record {record_number}"
        )
    hydrated_description = job_payload.get("description")
    if not isinstance(hydrated_description, str) or not hydrated_description.strip():
        raise DatasetValidationError(
            f"job_description_ref has no description: {job_path}"
        )
    return {**candidate, "job_description": hydrated_description}


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![\w+#]){re.escape(alias)}(?![\w+#])",
        re.IGNORECASE,
    )


COMPILED_SKILLS = tuple(
    (
        canonical,
        tuple(_alias_pattern(alias) for alias in aliases),
    )
    for canonical, aliases in SKILL_ALIASES.items()
)
COMPILED_LABEL_RULES = tuple(
    (
        label,
        confidence,
        tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns),
    )
    for label, confidence, patterns in LABEL_RULES
)


def _evidence_segments(description: str) -> list[str]:
    segments = re.split(r"(?<=[.!?])\s+|\n+", description)
    return [
        segment.strip()
        for segment in segments
        if 15 <= len(segment.strip()) <= 700
    ]


def _section_heading_label(text: str) -> tuple[str, float, str] | None:
    normalized = text.strip()
    for label, confidence, patterns in SECTION_LABEL_PATTERNS:
        for pattern in patterns:
            if pattern.fullmatch(normalized):
                return label, confidence, f"section:{pattern.pattern}"
    return None


def _evidence_segments_with_section_labels(
    description: str,
) -> list[tuple[str, tuple[str, float, str] | None]]:
    labelled: list[tuple[str, tuple[str, float, str] | None]] = []
    active_section: tuple[str, float, str] | None = None
    for raw_line in description.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading_label = _section_heading_label(line)
        if heading_label is not None:
            active_section = heading_label
            continue
        if SECTION_BREAK_PATTERN.fullmatch(line):
            active_section = None
            continue
        segments = re.split(r"(?<=[.!?])\s+", line)
        labelled.extend(
            (segment.strip(), active_section)
            for segment in segments
            if 15 <= len(segment.strip()) <= 700
        )
    return labelled


def _proposed_label(evidence: str) -> tuple[str, float, str] | None:
    for label, confidence, patterns in COMPILED_LABEL_RULES:
        for pattern in patterns:
            if pattern.search(evidence):
                return label, confidence, pattern.pattern
    return None


def _skills(evidence: str) -> list[str]:
    matched: list[str] = []
    for canonical, patterns in COMPILED_SKILLS:
        if any(pattern.search(evidence) for pattern in patterns):
            matched.append(canonical)
    # Prefer the more specific PySpark/Apache Spark distinction without
    # emitting both for the same PySpark token.
    if "PySpark" in matched and "Apache Spark" in matched:
        matched.remove("Apache Spark")
    return matched


def candidates_from_job(
    payload: dict[str, Any],
    role_catalog: RoleCatalog | None = None,
) -> list[dict[str, Any]]:
    job_id = payload.get("id")
    title = payload.get("title")
    description = payload.get("description")
    if not all(isinstance(value, str) and value.strip() for value in (job_id, title, description)):
        raise ValueError("processed job requires non-empty id, title, and description")

    metadata = payload.get("metadata")
    collection = metadata.get("collection") if isinstance(metadata, dict) else None
    collection = collection if isinstance(collection, dict) else {}
    role_family = collection.get("role_family")
    if not role_family and role_catalog is not None:
        role_family = role_catalog.match_title(title)

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for evidence, section_label in _evidence_segments_with_section_labels(
        description
    ):
        proposed = _proposed_label(evidence)
        if section_label is not None and (
            proposed is None or proposed[0] == "IMPORTANT"
        ):
            proposed = section_label
        if proposed is None:
            continue
        label, confidence, rule = proposed
        for requirement in _skills(evidence):
            key = (requirement.casefold(), evidence.casefold())
            if key in seen:
                continue
            seen.add(key)
            identity = json.dumps(
                {
                    "job_id": job_id,
                    "requirement": requirement.casefold(),
                    "evidence": evidence.casefold(),
                },
                sort_keys=True,
            )
            records.append(
                {
                    "candidate_id": hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest()[:20],
                    "annotation_status": "needs_review",
                    "reviewed_label": None,
                    "review_notes": "",
                    "job_id": job_id,
                    "job_title": title,
                    "job_description": description,
                    "requirement": requirement,
                    "proposed_label": label,
                    "evidence_span": evidence,
                    "proposal_confidence": confidence,
                    "proposal_rule": rule,
                    "source": payload.get("source"),
                    "source_url": payload.get("source_url"),
                    "role_family": role_family,
                    "collection_query": collection.get("query"),
                }
            )
    return records


def generate_candidate_queue(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    additional_candidate_paths: Iterable[str | Path] = (),
) -> CandidateGenerationSummary:
    jobs_read = 0
    jobs_failed = 0
    records: dict[str, dict[str, Any]] = {}
    try:
        role_catalog = RoleCatalog.load()
    except RoleCatalogError:
        role_catalog = None

    for raw_path in sorted(Path(path) for path in input_paths):
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("processed job must be a JSON object")
            jobs_read += 1
            for record in candidates_from_job(payload, role_catalog=role_catalog):
                record["job_description_ref"] = str(raw_path)
                records[record["candidate_id"]] = record
        except (OSError, json.JSONDecodeError, ValueError):
            jobs_failed += 1

    for additional_path in additional_candidate_paths:
        for record in read_candidate_records(additional_path):
            candidate_id = record.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise DatasetValidationError(
                    f"included candidate in {additional_path} requires candidate_id"
                )
            records.setdefault(candidate_id, record)

    ordered_records = [records[key] for key in sorted(records)]
    destination_path = Path(output_path)
    if destination_path.exists():
        previous_records = {
            record.get("candidate_id"): record
            for record in read_candidate_records(destination_path)
            if record.get("candidate_id")
        }
        for record in ordered_records:
            previous = previous_records.get(record["candidate_id"])
            if previous:
                record["annotation_status"] = previous.get(
                    "annotation_status",
                    record["annotation_status"],
                )
                record["reviewed_label"] = previous.get("reviewed_label")
                record["review_notes"] = previous.get("review_notes", "")
                for field_name in (
                    "requirement_group",
                    "group_operator",
                    "group_members",
                    "group_label",
                ):
                    if field_name in previous:
                        record[field_name] = previous[field_name]
    output_records = (
        [_compact_review_record(record) for record in ordered_records]
        if Path(output_path).suffix.casefold() == ".json"
        else ordered_records
    )
    destination = write_candidate_records(output_records, output_path)
    label_counts = Counter(
        record["proposed_label"]
        for record in ordered_records
    )
    return CandidateGenerationSummary(
        jobs_read=jobs_read,
        jobs_failed=jobs_failed,
        candidates=len(ordered_records),
        label_counts=dict(sorted(label_counts.items())),
        output_path=destination,
    )


def validate_candidate_reviews(
    candidate_path: str | Path,
    output_path: str | Path,
) -> CandidateReviewValidationSummary:
    """Audit structured decisions and note/label disagreements without editing them."""
    source = Path(candidate_path)
    records = read_candidate_records(source)
    issues: list[dict[str, Any]] = []

    def add_issue(
        severity: str,
        code: str,
        record_number: int,
        candidate: dict[str, Any],
        message: str,
        **details: Any,
    ) -> None:
        issues.append(
            {
                "severity": severity,
                "code": code,
                "record_number": record_number,
                "candidate_id": candidate.get("candidate_id"),
                "requirement": candidate.get("requirement"),
                "message": message,
                **details,
            }
        )

    for record_number, candidate in enumerate(records, start=1):
        raw_status = candidate.get("annotation_status")
        try:
            status = normalize_annotation_status(raw_status)
        except DatasetValidationError as exc:
            add_issue(
                "error",
                "invalid_status",
                record_number,
                candidate,
                str(exc),
            )
            continue
        if raw_status != status:
            add_issue(
                "information",
                "status_normalized",
                record_number,
                candidate,
                f"{raw_status!r} is interpreted as {status!r}.",
                normalized_status=status,
            )

        try:
            proposed_label = _normalize_label(
                candidate.get("proposed_label"),
                "proposed_label",
            )
            reviewed_label = _normalize_label(
                candidate.get("reviewed_label"),
                "reviewed_label",
            )
        except DatasetValidationError as exc:
            add_issue(
                "error",
                "invalid_label",
                record_number,
                candidate,
                str(exc),
            )
            continue

        notes = candidate.get("review_notes")
        if notes is not None and not isinstance(notes, str):
            add_issue(
                "error",
                "invalid_review_notes",
                record_number,
                candidate,
                "review_notes must be a string or null.",
            )
            continue
        mentioned_labels = _labels_mentioned_in_notes(notes or "")
        effective_label = reviewed_label or proposed_label

        if status == "approved" and effective_label is None:
            add_issue(
                "error",
                "approved_without_label",
                record_number,
                candidate,
                "Approved candidate has neither reviewed_label nor proposed_label.",
            )
        elif status == "approved" and reviewed_label is None:
            add_issue(
                "information",
                "proposed_label_accepted",
                record_number,
                candidate,
                f"Approval accepts the proposed {proposed_label} label.",
                effective_label=effective_label,
            )

        if status == "approved" and effective_label is not None:
            try:
                hydrated = _hydrate_job_description(
                    candidate,
                    candidate_path=source,
                    record_number=record_number,
                )
                RequirementExample.from_dict(
                    {
                        **hydrated,
                        "label": effective_label,
                        "annotation_source": "human_review",
                    }
                )
            except DatasetValidationError as exc:
                add_issue(
                    "error",
                    "invalid_approved_candidate",
                    record_number,
                    candidate,
                    str(exc),
                )
                continue

        alternatives = [
            label for label in mentioned_labels if label != effective_label
        ]
        if status == "rejected" and len(mentioned_labels) == 1:
            suggested_label = mentioned_labels[0]
            add_issue(
                "warning",
                "possible_label_correction_rejected",
                record_number,
                candidate,
                "The note names a label for a rejected candidate. If the extraction "
                "is valid, approve it and put this label in reviewed_label.",
                suggested_status="approved",
                suggested_reviewed_label=suggested_label,
            )
        elif status == "approved" and reviewed_label is None and alternatives:
            add_issue(
                "warning",
                "note_label_disagrees_with_effective_label",
                record_number,
                candidate,
                "The note mentions alternative labels; confirm reviewed_label captures "
                "the intended final decision.",
                effective_label=effective_label,
                labels_mentioned_in_notes=mentioned_labels,
            )

    counts = Counter(issue["severity"] for issue in issues)
    payload = {
        "summary": {
            "candidates_read": len(records),
            "errors": counts["error"],
            "warnings": counts["warning"],
            "information": counts["information"],
        },
        "semantics": {
            "approved_without_reviewed_label": "accept proposed_label",
            "approved_with_reviewed_label": "use reviewed_label as the final label",
            "rejected": "exclude because the extracted requirement/evidence is invalid",
            "review_notes": "preserve reviewer reasoning; never silently infer a final label",
        },
        "issues": issues,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return CandidateReviewValidationSummary(
        candidates_read=len(records),
        errors=counts["error"],
        warnings=counts["warning"],
        information=counts["information"],
        output_path=destination,
    )


def promote_approved_candidates(
    candidate_path: str | Path,
    output_path: str | Path,
    decision_log_path: str | Path | None = None,
) -> CandidatePromotionSummary:
    source = Path(candidate_path)

    approved_examples: dict[str, RequirementExample] = {}
    conflict_labels: dict[tuple[str, str, str], str] = {}
    candidates_read = 0
    rejected = 0
    pending = 0
    decisions: list[dict[str, Any]] = []
    for record_number, candidate in enumerate(
        read_candidate_records(source),
        start=1,
    ):
        candidates_read += 1
        raw_status = candidate.get("annotation_status")
        try:
            status = normalize_annotation_status(raw_status)
            proposed_label = _normalize_label(
                candidate.get("proposed_label"),
                "proposed_label",
            )
            reviewed_label = _normalize_label(
                candidate.get("reviewed_label"),
                "reviewed_label",
            )
        except DatasetValidationError as exc:
            raise DatasetValidationError(
                f"{source} record {record_number}: {exc}"
            ) from exc
        notes = candidate.get("review_notes")
        if notes is not None and not isinstance(notes, str):
            raise DatasetValidationError(
                f"review_notes at {source} record {record_number} must be a string"
            )
        effective_label = reviewed_label
        label_source = "reviewed_label" if reviewed_label else None
        if status == "approved" and effective_label is None:
            effective_label = proposed_label
            label_source = "proposed_label"
        decisions.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "annotation_status": status,
                "annotation_status_raw": raw_status,
                "reviewed_label": reviewed_label,
                "effective_label": effective_label,
                "label_source": label_source,
                "review_notes": notes or "",
                "proposed_label": proposed_label,
                "requirement": candidate.get("requirement"),
                "evidence_span": candidate.get("evidence_span"),
                "proposal_rule": candidate.get("proposal_rule"),
                "job_id": candidate.get("job_id"),
                "job_title": candidate.get("job_title"),
                "role_family": candidate.get("role_family"),
                "source": candidate.get("source"),
                "is_synthetic": bool(candidate.get("is_synthetic")),
                "requirement_group": candidate.get("requirement_group"),
                "group_operator": candidate.get("group_operator"),
                "group_members": candidate.get("group_members"),
                "group_label": candidate.get("group_label"),
            }
        )
        if status == "rejected":
            rejected += 1
            continue
        if status != "approved":
            pending += 1
            continue
        candidate = _hydrate_job_description(
            candidate,
            candidate_path=source,
            record_number=record_number,
        )

        if effective_label is None:
            raise DatasetValidationError(
                f"approved candidate at {source} record {record_number} "
                "requires reviewed_label or proposed_label"
            )
        if (
            candidate.get("is_synthetic") is True
            and str(candidate.get("synthetic_type", "")).startswith(
                "explicit_negation:"
            )
            and effective_label != "NOT_REQUIREMENT"
        ):
            raise DatasetValidationError(
                f"explicit-negation candidate at {source} record {record_number} must "
                "be approved as NOT_REQUIREMENT or left unapproved"
            )
        try:
            example = RequirementExample.from_dict(
                {
                    **candidate,
                    "label": effective_label,
                    "annotation_source": "human_review",
                }
            )
        except DatasetValidationError as exc:
            raise DatasetValidationError(
                f"{source} record {record_number}: {exc}"
            ) from exc

        previous_label = conflict_labels.get(example.conflict_key)
        if previous_label is not None and previous_label != example.label:
            raise DatasetValidationError(
                f"conflicting approved labels at {source} record {record_number}: "
                f"{previous_label} and {example.label}"
            )
        conflict_labels[example.conflict_key] = example.label
        approved_examples[example.example_id] = example

    decision_destination = None
    if decision_log_path is not None:
        decision_destination = _write_review_decision_log(
            decisions,
            decision_log_path,
        )
    if not approved_examples:
        raise DatasetValidationError("candidate queue has no approved examples")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "job_id": example.job_id,
            "job_title": example.job_title,
            "job_description": example.job_description,
            "requirement": example.requirement,
            "label": example.label,
            "evidence_span": example.evidence_span,
            "annotation_source": example.annotation_source,
            "review_notes": example.review_notes,
            "role_family": example.role_family,
            "source": example.source,
            "source_url": example.source_url,
            "is_synthetic": example.is_synthetic,
            "synthetic_type": example.synthetic_type,
            "source_example_id": example.source_example_id,
            "source_label": example.source_label,
            "original_evidence_span": example.original_evidence_span,
            "generation_seed": example.generation_seed,
            "requirement_group": example.requirement_group,
            "group_operator": example.group_operator,
            "group_members": list(example.group_members),
            "group_label": example.group_label,
        }
        for example in sorted(
            approved_examples.values(),
            key=lambda item: item.example_id,
        )
    ]
    destination.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return CandidatePromotionSummary(
        candidates_read=candidates_read,
        approved=len(records),
        rejected=rejected,
        pending=pending,
        output_path=destination,
        decision_log_path=decision_destination,
    )


def _write_review_decision_log(
    decisions: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    status_counts = Counter(
        str(decision.get("annotation_status") or "needs_review")
        for decision in decisions
    )
    rejected_rules = Counter(
        str(decision.get("proposal_rule") or "unknown")
        for decision in decisions
        if decision.get("annotation_status") == "rejected"
    )
    label_corrections = Counter(
        f"{decision.get('proposed_label')} -> {decision.get('effective_label')}"
        for decision in decisions
        if decision.get("annotation_status") == "approved"
        and decision.get("effective_label")
        and decision.get("effective_label") != decision.get("proposed_label")
    )
    payload = {
        "summary": {
            "total": len(decisions),
            "with_review_notes": sum(
                bool(decision.get("review_notes")) for decision in decisions
            ),
            "accepted_proposed_labels": sum(
                decision.get("label_source") == "proposed_label"
                for decision in decisions
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "rejected_by_proposal_rule": dict(sorted(rejected_rules.items())),
            "label_corrections": dict(sorted(label_corrections.items())),
        },
        "decisions": decisions,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
