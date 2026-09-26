"""Small, deterministic drafting checks; not semantic proof or an ATS score."""

from app.applications.models import TailoredCV


TAILORING_POLICY = {
    "target_pages": 2,
    "target_words": "450–600 when sufficient relevant source material exists; shorter is fine",
    "max_words": 650,
    "max_summary_words": 70,
    "max_skill_entries": 5,
    "max_noncontact_entries": 32,
    "primary_role_achievement_bullets": "6–8 selected achievements, not every source bullet",
    "earlier_role_achievement_bullets": "2–3 per role, while retaining employment chronology",
    "relevant_projects": "1–2 strongest supplied projects when relevant; label independent work accurately",
}


def cv_quality_metrics(cv: TailoredCV) -> dict[str, int]:
    return {
        "word_count": sum(len(entry.text.split()) for section in cv.sections for entry in section.entries),
        "summary_words": sum(len(entry.text.split()) for section in cv.sections
                             if section.heading == "Professional Summary" for entry in section.entries),
        "skill_entries": sum(len(section.entries) for section in cv.sections if section.heading == "Skills"),
        "noncontact_entries": sum(len(section.entries) for section in cv.sections if section.heading != "Contact"),
    }


def validate_tailoring(cv: TailoredCV) -> list[str]:
    """Enforce bounded presentation, leaving factual support to separate checks.

    A word/entry limit cannot guarantee two rendered pages. The exported PDF
    still needs a layout check, and short unsupported claims remain unsupported.
    """
    metrics = cv_quality_metrics(cv)
    errors = []
    for metric, policy_key, label in (
        ("word_count", "max_words", "CV word count"),
        ("summary_words", "max_summary_words", "Summary word count"),
        ("skill_entries", "max_skill_entries", "Skills entry count"),
        ("noncontact_entries", "max_noncontact_entries", "Non-contact entry count"),
    ):
        limit = TAILORING_POLICY[policy_key]
        if metrics[metric] > limit:
            errors.append(f"{label} {metrics[metric]} exceeds the tailoring limit of {limit}.")
    return errors
