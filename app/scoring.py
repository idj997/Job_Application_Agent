"""Reproducible decision rules; model evidence is validated before it is used."""
from __future__ import annotations

import re

from app.applications.models import CVEvaluation, Decision, MatchAssessment, TailoredCV


WEIGHTS = {"CORE": 5, "IMPORTANT": 3, "PREFERRED": 1, "CONTEXTUAL": 0, "NOT_REQUIREMENT": 0}


def normalized(text: str) -> str:
    return " ".join(text.split())


def is_quote(quote: str, source: str) -> bool:
    return bool(normalized(quote)) and normalized(quote) in normalized(source)


def decide(
    assessment: MatchAssessment,
    cv_text: str,
    job_text: str,
    constraint_ids: set[str],
    threshold: int = 70,
) -> Decision:
    errors = []
    total = sum(WEIGHTS[item.importance] for item in assessment.requirements)
    earned = sum(
        WEIGHTS[item.importance] * {"MET": 1, "PARTIAL": 0.5, "MISSING": 0, "UNKNOWN": 0}[item.status]
        for item in assessment.requirements
    )
    score = round(100 * earned / total) if total else 0
    if not assessment.description_complete:
        errors.append("The job description is incomplete.")
    if not total:
        errors.append("No scored job requirements were extracted.")
    names = [item.requirement.casefold() for item in assessment.requirements]
    if len(names) != len(set(names)):
        errors.append("Duplicate requirements would distort the match score.")
    for item in assessment.requirements:
        if not is_quote(item.job_evidence, job_text):
            errors.append(f"Job evidence could not be verified: {item.requirement}.")
        if any(not is_quote(quote, cv_text) for quote in item.cv_evidence):
            errors.append(f"CV evidence could not be verified: {item.requirement}.")
        if item.status in {"MET", "PARTIAL"} and not item.cv_evidence:
            errors.append(f"Supporting CV evidence is missing: {item.requirement}.")
    actual_ids = [item.constraint_id for item in assessment.constraint_checks]
    if set(actual_ids) != constraint_ids or len(actual_ids) != len(set(actual_ids)):
        errors.append("The assessment did not check every configured constraint exactly once.")
    for item in assessment.constraint_checks:
        if item.status != "UNKNOWN" and not item.job_evidence:
            errors.append(f"Constraint verdict has no job evidence: {item.constraint_id}.")
        if item.job_evidence and not is_quote(item.job_evidence, job_text):
            errors.append(f"Constraint evidence could not be verified: {item.constraint_id}.")
    if errors:
        return Decision(verdict="REVIEW", score=score, reasons=errors)
    failures = [item.explanation for item in assessment.constraint_checks if item.status == "FAIL"]
    if failures:
        return Decision(verdict="SKIP", score=score, reasons=failures)
    unknowns = [item.explanation for item in assessment.constraint_checks if item.status == "UNKNOWN"]
    core_gaps = [item.requirement for item in assessment.requirements if item.importance == "CORE" and item.status != "MET"]
    if unknowns or core_gaps:
        return Decision(
            verdict="REVIEW", score=score,
            reasons=unknowns + (["Unresolved mandatory requirements: " + ", ".join(core_gaps)] if core_gaps else []),
        )
    if assessment.recommended_verdict != "APPLY":
        return Decision(verdict=assessment.recommended_verdict, score=score, reasons=[assessment.rationale])
    if score < threshold:
        return Decision(verdict="SKIP", score=score, reasons=[f"Match score {score} is below your threshold of {threshold}."])
    return Decision(verdict="APPLY", score=score, reasons=[assessment.rationale])


def validate_cv(cv: TailoredCV, master_cv: str) -> list[str]:
    errors = []
    headings = [section.heading for section in cv.sections]
    if not headings or headings[0] != "Contact":
        errors.append("The tailored CV must start with contact details from the master CV.")
    if len(headings) != len(set(headings)):
        errors.append("The tailored CV contains duplicate sections.")
    for section in cv.sections:
        for entry in section.entries:
            if "\n" in entry.text or "\r" in entry.text or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", entry.text):
                errors.append("Each CV entry must be one plain-text line without control characters.")
            if any(not is_quote(quote, master_cv) for quote in entry.evidence_quotes):
                errors.append(f"CV source evidence could not be verified for: {entry.text}")
    return errors


def cv_score(evaluation: CVEvaluation) -> int:
    return round(evaluation.requirement_coverage * 0.7 + evaluation.clarity * 0.3)


def cv_ready(evaluation: CVEvaluation, threshold: int) -> bool:
    return (
        evaluation.factual_consistency
        and not evaluation.unsupported_claims
        and not evaluation.missing_critical_information
        and cv_score(evaluation) >= threshold
    )
