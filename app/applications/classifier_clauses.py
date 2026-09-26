"""Narrow, qualifier-preserving decomposition of shared skill lists.

This is a grammar helper, not a source of evidence or a suitability decision.
Atoms are derived hypotheses; callers must retain the unchanged job quote and
must not mark a group MET when ``uncertainty`` is set. Unsupported syntax returns
None, so the matching layer can abstain instead of silently dropping qualifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


@dataclass(frozen=True)
class SkillDecomposition:
    atoms: tuple[str, ...]
    combination: Literal["all", "any"]
    uncertainty: str | None = None


# Only front-loaded shared qualifiers are distributed. Rear-loaded phrases such
# as "professional Python or SQL experience" remain deliberately unsupported.
_PREFIX = re.compile(
    r"^(?P<prefix>(?:(?:proven|demonstrated|hands[- ]on|professional|commercial|"
    r"industry|production|practical|working|strong|extensive|solid|deep|advanced|"
    r"expert|basic|good|excellent)\s+)*(?:experience\s+(?:with|in|of)|"
    r"knowledge\s+of|familiarity\s+with|understanding\s+of|proficiency\s+in))\s+"
    r"(?P<body>.+)$", re.I,
)
_UNSAFE = re.compile(
    r"\d|[;:!?()\[\]{}/=<>]|\b(?:not|no|never|without|unless|except|if|"
    r"either|equivalent|instead|only|including|such\s+as|e\.g|i\.e|"
    r"years?|months?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"degree|bachelor\w*|master\w*|bsc|msc|phd|doctorate|certifi\w*|"
    r"citizens?\w*|nationality|eligible|eligibility|clearance|licen[cs]e|"
    r"visa|sponsorship|authori[sz]ation|fluen\w*|native|preferred|desirable|"
    r"bonus|optional|advantageous|minimum|least|more|less)\b", re.I,
)
_ITEM_UNSAFE = re.compile(
    r"\b(?:in|on|for|with|from|of|at|as|through|under|by|using|to|"
    r"is|are|has|have|must|should|will|can|ability|experience|knowledge|"
    r"familiarity|understanding|proficiency|skills?|required|essential|mandatory|"
    r"proven|demonstrated|hands[- ]on|professional|commercial|industry|production|"
    r"practical|working|strong|extensive|solid|deep|advanced|expert|basic|good|"
    r"excellent|other|related|similar|etc)\b", re.I,
)
_ITEM_WORD = r"[A-Za-z][A-Za-z+#-]*(?:\.[A-Za-z][A-Za-z+#-]*)*"
_ITEM = re.compile(rf"{_ITEM_WORD}(?:\s+{_ITEM_WORD}){{0,4}}\Z")
_SEPARATOR = re.compile(r",\s*(?:and|or)\b|,|\band\b|\bor\b|&", re.I)


def decompose_shared_skills(text: str) -> SkillDecomposition | None:
    """Parse a simple shared-prefix list without weakening its requirements.

    For example, ``Hands-on experience with Python and SQL`` produces two
    hypotheses beginning with ``Hands-on experience with``. Comma lists are
    conjunctions unless an unambiguous final ``or`` specifies alternatives.
    Nested/mixed coordination, trailing shared qualifiers, numeric experience,
    credentials and eligibility clauses are intentionally outside this grammar.

    An open-ended ``etc.`` list still exposes its explicit members for partial
    matching, but its unresolved remainder is recorded and prevents full MET.
    """
    if not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text:
        return None
    claim = text.strip()
    if claim.endswith("."):
        claim = claim[:-1].strip()
    claim = re.sub(r"\s+(?:is|are)\s+(?:required|essential|mandatory)$", "", claim, flags=re.I)
    if _UNSAFE.search(claim):
        return None
    match = _PREFIX.fullmatch(claim)
    if not match:
        return None
    prefix, body = match.group("prefix"), match.group("body")
    open_ended = bool(re.search(r",\s*(?:and\s+)?etc\.?$", body, re.I))
    if open_ended:
        body = re.sub(r",\s*(?:and\s+)?etc\.?$", "", body, flags=re.I).strip()
    separators = list(_SEPARATOR.finditer(body))
    if not separators:
        return None
    pieces = []
    relations = set()
    word_connector_seen = False
    previous = 0
    for separator in separators:
        pieces.append(body[previous:separator.start()].strip())
        connector = separator.group().casefold()
        if connector == ",":
            # A comma after "or" can change the grouping: do not guess which
            # alternatives belong together (same conservative rule for "and").
            if word_connector_seen:
                return None
        else:
            word_connector_seen = True
            relations.add("any" if re.search(r"\bor\b", connector) else "all")
        previous = separator.end()
    pieces.append(body[previous:].strip())
    if len(relations) > 1 or not 2 <= len(pieces) <= 8:
        return None
    if any(not _ITEM.fullmatch(piece) or _ITEM_UNSAFE.search(piece) for piece in pieces):
        return None
    # Empty or repeated members can be a malformed list, not multiple evidence
    # obligations. Reject instead of changing its structure by deduplicating.
    if len({piece.casefold() for piece in pieces}) != len(pieces):
        return None
    combination = next(iter(relations), "all")
    return SkillDecomposition(
        atoms=tuple(f"{prefix} {piece}" for piece in pieces),
        combination=combination,
        uncertainty=(
            "The open-ended skill list has an unspecified remainder; explicit members can be matched but complete coverage requires review."
            if open_ended else None
        ),
    )
