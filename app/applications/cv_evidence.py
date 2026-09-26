"""Resolve model-selected source IDs into exact, locally supplied CV evidence.

The source catalog retains every nonblank source line without summarizing or
truncating it. IDs establish traceability only: selecting a real line does not
prove that the generated claim is supported by that line. The ordinary CV
validation and factual review still apply after resolution.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StrictStr, StringConstraints, ValidationError, field_validator

from app.applications.models import CVEntry, CVSection, StrictModel, TailoredCV


SourceID = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=False, pattern=r"^S[0-9]{4,}$"),
]


class _ReferenceModel(StrictModel):
    # Validation messages may be recorded in a workflow ledger. Do not echo
    # source text or arbitrary model-supplied values into those messages.
    model_config = ConfigDict(hide_input_in_errors=True)


class CVSource(_ReferenceModel):
    source_id: SourceID
    text: StrictStr = Field(min_length=1)


class ReferencedCVEntry(_ReferenceModel):
    text: StrictStr = Field(min_length=1)
    style: Literal["paragraph", "bullet"]
    evidence_ids: list[SourceID] = Field(min_length=1, strict=True)

    @field_validator("evidence_ids")
    @classmethod
    def require_unique_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("CV entry contains duplicate evidence IDs.")
        return values


class ReferencedCVSection(_ReferenceModel):
    heading: Literal[
        "Contact", "Professional Summary", "Experience", "Education", "Skills",
        "Projects", "Certifications", "Languages", "Publications", "Volunteering",
    ]
    entries: list[ReferencedCVEntry] = Field(min_length=1)


class ReferencedCV(_ReferenceModel):
    sections: list[ReferencedCVSection] = Field(min_length=1)


def build_cv_sources(master_cv: str) -> list[CVSource]:
    """Number every nonblank line in order; edge whitespace alone is stripped.

    Repeated source lines retain separate IDs. There is deliberately no line
    count or line-length truncation: a provider's input budget must reject an
    oversized complete prompt instead of silently discarding CV evidence.
    """
    if not isinstance(master_cv, str):
        raise TypeError("Master CV must be a string.")
    lines = [line.strip() for line in master_cv.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Master CV contains no nonblank source lines.")
    return [
        CVSource(source_id=f"S{index:04d}", text=line)
        for index, line in enumerate(lines, start=1)
    ]


def resolve_cv_references(cv: ReferencedCV, sources: list[CVSource]) -> TailoredCV:
    """Convert validated IDs to exact source lines, never model-written quotes.

    Callers supply the catalog from ``build_cv_sources(master_cv)`` for this
    generation. Unknown/duplicate IDs or invalid/mutated inputs fail closed;
    errors identify the problem without including generated or source text.
    """
    if not isinstance(cv, ReferencedCV):
        raise ValueError("CV references must use the referenced CV schema.")
    if not isinstance(sources, list) or not sources:
        raise ValueError("CV source catalog must be a nonempty list.")

    # Revalidate serialized models: Pydantic objects can have been mutated or
    # created via model_construct(), neither of which necessarily validates.
    try:
        checked_cv = ReferencedCV.model_validate(cv.model_dump(warnings=False))
    except (ValidationError, TypeError, ValueError):
        raise ValueError("CV references do not satisfy the referenced CV schema.") from None

    catalog: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, CVSource):
            raise ValueError("CV source catalog contains an invalid source entry.")
        try:
            checked_source = CVSource.model_validate(source.model_dump(warnings=False))
        except (ValidationError, TypeError, ValueError):
            raise ValueError("CV source catalog contains an invalid source entry.") from None
        if checked_source.source_id in catalog:
            raise ValueError("CV source catalog contains duplicate source IDs.")
        catalog[checked_source.source_id] = checked_source.text

    sections = []
    for section_index, section in enumerate(checked_cv.sections, start=1):
        entries = []
        for entry_index, entry in enumerate(section.entries, start=1):
            if any(source_id not in catalog for source_id in entry.evidence_ids):
                raise ValueError(
                    f"CV section {section_index}, entry {entry_index} references an unknown source ID."
                )
            entries.append(CVEntry(
                text=entry.text,
                style=entry.style,
                evidence_quotes=[catalog[source_id] for source_id in entry.evidence_ids],
            ))
        sections.append(CVSection(heading=section.heading, entries=entries))
    return TailoredCV(sections=sections)


def reconcile_exact_cv_references(
    cv: ReferencedCV, sources: list[CVSource],
) -> tuple[ReferencedCV, list[dict]]:
    """Relink verbatim source-line entries, recording only reference changes.

    Original references must already be valid: this cannot rescue an unknown,
    malformed, or duplicate ID. Matching uses complete string equality, never
    whitespace normalization, punctuation changes, substrings, or semantics.
    An existing exact citation is retained; otherwise the first catalog line
    with identical text replaces the selected IDs. No generated text is edited.

    Returns a deep copy and non-PII notes with one-based section/entry indices.
    Correcting a verbatim citation does not establish broader factual accuracy.
    """
    resolve_cv_references(cv, sources)
    reconciled = cv.model_copy(deep=True)
    catalog = {source.source_id: source.text for source in sources}
    first_exact_id: dict[str, str] = {}
    for source in sources:
        first_exact_id.setdefault(source.text, source.source_id)

    notes = []
    for section_index, section in enumerate(reconciled.sections, start=1):
        for entry_index, entry in enumerate(section.entries, start=1):
            exact_id = first_exact_id.get(entry.text)
            if exact_id is None or any(catalog[source_id] == entry.text for source_id in entry.evidence_ids):
                continue
            original_ids = list(entry.evidence_ids)
            entry.evidence_ids = [exact_id]
            notes.append({
                "section_index": section_index,
                "entry_index": entry_index,
                "original_ids": original_ids,
                "resolved_ids": [exact_id],
                "reason": "exact_source_line_match",
            })
    return reconciled, notes
