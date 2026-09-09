from typing import Literal


ImportanceCategory = Literal[
    "core",
    "preferred",
    "contextual",
    "ambiguous",
]


CORE_TERMS = [
    "required",
    "essential",
    "must have",
    "must-have",
    "mandatory",
    "strong experience",
    "strong knowledge",
    "proven experience",
]

PREFERRED_TERMS = [
    "desirable",
    "preferred",
    "nice to have",
    "nice-to-have",
    "advantageous",
    "bonus",
    "would be useful",
    "would be beneficial",
]

CONTEXTUAL_TERMS = [
    "exposure to",
    "awareness of",
    "familiarity with",
    "understanding of",
]


def classify_importance(skill: dict) -> dict:
    """
    Classify one extracted skill using only explicit wording
    in the local sentence/context.

    The classifier deliberately returns 'ambiguous' when
    wording is not explicit enough.
    """

    context = skill["context"].lower()

    if any(term in context for term in CORE_TERMS):
        category: ImportanceCategory = "core"
        rule_score = 1.0
        matched_rule = "core_term"

    elif any(term in context for term in PREFERRED_TERMS):
        category = "preferred"
        rule_score = 0.6
        matched_rule = "preferred_term"

    elif any(term in context for term in CONTEXTUAL_TERMS):
        category = "contextual"
        rule_score = 0.3
        matched_rule = "contextual_term"

    else:
        category = "ambiguous"
        rule_score = None
        matched_rule = None

    return {
        **skill,
        "importance_category": category,
        "rule_score": rule_score,
        "matched_rule": matched_rule,
    }


def classify_skills(skills: list[dict]) -> list[dict]:
    return [
        classify_importance(skill)
        for skill in skills
    ]