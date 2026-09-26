"""Conservative classifier-led matching using quoted requirements and CV evidence.

This is a pretrained general NLI baseline with explicit rules, not a classifier
trained or calibrated on this project's candidate/job decisions. No generative
provider is used. Unresolved coverage or mandatory reasoning needs REVIEW.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from app.applications.models import CandidateProfile, MatchAssessment, RequirementMatch
from app.applications.classifier_clauses import decompose_shared_skills
from app.classifiers.importance_rule import classify_importance
from app.classifiers.requirement_nli import NLIInputTooLong, RequirementNLIClassifier
from app.providers.base import ModelProviderError

MAX_REQUIREMENTS = 64
MAX_CV_UNITS = 180
MAX_WINDOW_CHARS = 640
MAX_CANDIDATES = 8
MAX_PAIRS = 384
ENTAILMENT_THRESHOLD = 0.90  # An operating rule, not a calibrated probability of suitability.

_HEADINGS = {
    "requirements": "CORE", "required skills": "CORE", "essential skills": "CORE",
    "minimum requirements": "CORE", "minimum qualifications": "CORE", "qualifications": "CORE",
    "essential": "CORE", "what you need": "CORE", "must have": "CORE",
    "preferred qualifications": "PREFERRED", "preferred skills": "PREFERRED",
    "desirable": "PREFERRED", "preferred": "PREFERRED", "nice to have": "PREFERRED",
    "responsibilities": "IMPORTANT", "your responsibilities": "IMPORTANT",
    "what you will do": "IMPORTANT", "what you'll do": "IMPORTANT", "skills": "IMPORTANT",
    "about you": "IMPORTANT", "what you bring": "IMPORTANT",
    "what we're looking for": "CORE", "what we are looking for": "CORE",
    "benefits": None, "what we offer": None, "about us": None, "about the company": None,
    "equal opportunities": None, "how to apply": None,
    "description": None, "job description": None, "about the role": None,
    "role overview": None, "overview": None,
}
_NARRATIVE_HEADINGS = {"description", "job description", "about the role", "role overview", "overview"}
_EXPLICIT_CORE = re.compile(r"\b(required|essential|mandatory|must|minimum)\b", re.I)
_PREFERRED = re.compile(r"\b(preferred|desirable|bonus|advantageous)\b|nice[- ]to[- ]have|would be (?:useful|beneficial)|\b(?:is|are|would be)\s+(?:a\s+)?(?:great\s+|strong\s+)?plus\b", re.I)
_MARKER = re.compile(
    r"\b(require[ds]?|requirements?|essential|mandatory|must|qualification[s]?|experience|"
    r"skills?|knowledge|familiarity|understanding|proficien\w*|ability|able to|degree|"
    r"bachelor\w*|master\w*|bsc|msc|phd|doctorate|certifi\w*|years?|fluen\w*|"
    r"competent|comfortable|strong|excellent|desirable|preferred)\b|you (?:will|have|should)", re.I,
)
_DUTY = re.compile(r"^(?:you will |you'll |to )?(build|develop|design|maintain|deploy|write|collaborate|lead|support|manage|test|analy[sz]e|troubleshoot|implement|deliver|create)\b", re.I)
_NEGATIVE = re.compile(r"\b(no|not|never|without|lack|lacking|missing|cannot)\b|\b(?:haven|hasn|hadn|can|don|doesn|didn|isn|aren|wasn|weren|won|wouldn|couldn|shouldn)['’]t\b", re.I)
_ASPIRATIONAL = re.compile(r"\b(want to|hope to|planning to|learning|studying|pursuing|in progress|enrolled|beginner in|interested in|wish to|aim to)\b", re.I)
_EXPERIENCE = re.compile(r"\b(experience|professional|commercial|production|hands[- ]on|proven)\b", re.I)
_PROFESSIONAL = re.compile(r"\b(professional|commercial|industry|paid employment)\b", re.I)
_PROVENANCE = re.compile(r"\b(professional|commercial|employed|employment|employer)\b|\b(?:at work|as part of my job|paid client)\b", re.I)
_NONPROFESSIONAL = re.compile(r"\b(coursework|course work|academic|university projects?|student projects?|course projects?|side projects?|personal projects?|hobby|hackathons?|tutorials?|simulated client)\b", re.I)
_ELIGIBILITY = re.compile(r"\b(citizens?|citizenship|nationals?|nationality|eligible|eligibility|clearance|licen[cs]e|fluent|fluency)\b|\bright to work\b|\bwork (?:authori[sz]ation|permit)\b|\bnative (?:speaker|language)\b", re.I)
_PAST_ACTIONS = r"built|developed|implemented|designed|maintained|deployed|delivered|managed|led|worked|supported|automated|created|tested|programmed|engineered|analysed|analyzed|used|contributed|extended|wrote|tuned|compared|integrated|applied|evaluated"
_ACTION = re.compile(r"\b(" + _PAST_ACTIONS + r"|experience with|experience in|professional experience|commercial experience)\b", re.I)
_SUBJECTLESS_FRAME = re.compile(r"^(?:contributed to|worked (?:on|with|as|at|in))\s+", re.I)
_FIRST_PERSON_ACTION = re.compile(r"^I\s+(?:have\s+)?(?:" + _PAST_ACTIONS + r")\b", re.I)
_OTHER_ACTOR = re.compile(r"\b(he|she|they|colleagues?|coworkers?|co-workers?|team members?|supervisors?|mentors?|(?:my|our|the) team)\b", re.I)
_NAMED_OTHER_ACTION = re.compile(r"^(?!I\b|The candidate\b)[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s+(?:has\s+)?(?:" + _PAST_ACTIONS + r")\b")
_AMBIGUOUS_ACHIEVEMENT = re.compile(
    r"\?|\b(can|could|would|should|may|might|must|shall|will|if|unless|who|whose|"
    r"was|were|is|are|been|include[sd]?|contain[sd]?|require[sd]?|provide[sd]?|"
    r"allow[sd]?|enable[sd]?|refer[sd]?|remain[sd]?|seem[sd]?|appear[sd]?|"
    r"improve[sd]?|replace[sd]?|offer[sd]?|exist[sd]?|belong[sd]?|mean[sd]?)\b", re.I,
)
_DEGREE = re.compile(r"\b(bsc|msc|beng|meng|mba|phd|undergraduate|degree|bachelor\w*|master\w*|doctorate|certifi\w*)\b", re.I)
_QUANTITY = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|years?|months?|at least|more than|minimum of)\b", re.I)
_COMPLEX = re.compile(r"\b(unless|except|equivalent|either|only if|instead of|not only)\b", re.I)
_STOP = set("the a an of for to in on at and or with using is are be been has have must should will would can could you your our we us candidate skill skills experience knowledge familiarity understanding proven strong required essential mandatory preferred desirable professional commercial practical working ability able responsibility responsibilities requirement requirements have needs need minimum excellent".split())
_CANONICAL = {"bsc": "bachelor", "bachelors": "bachelor", "msc": "master", "masters": "master", "phd": "doctorate", "built": "build", "developed": "develop", "designed": "design", "managed": "manage", "created": "create", "deployed": "deploy", "implemented": "implement", "tested": "test"}
_CANONICAL.update({"optimisation": "optimization", "optimizations": "optimization",
                   "optimisations": "optimization", "statistical": "statistics"})


@dataclass(frozen=True)
class _Unit:
    text: str
    quote: str
    importance: str
    ambiguous: str | None = None


def _spans(text: str) -> list[tuple[str, int, int]]:
    """Keep source offsets; do not truncate sentences or split decimal numbers."""
    result = []
    for line in re.finditer(r"[^\n\r]+", text):
        bullet = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line.group())
        offset = bullet.end() if bullet else 0
        for sentence in re.finditer(r".+?(?:[.!?;](?=\s|$)|$)", line.group()[offset:]):
            raw = sentence.group()
            clean = re.sub(r"^\s*[-*•]\s+", "", raw).strip()
            if clean:
                start = line.start() + offset + sentence.start() + raw.find(clean)
                result.append((clean, start, start + len(clean)))
    return result


def _tokens(text: str) -> set[str]:
    tokens = {item.strip(".-") for item in re.findall(r"[a-z0-9][a-z0-9+#.-]*", text.casefold())}
    # 'Python-based' explicitly names its basis; '-like' or arbitrary fuzzy
    # neighbours do not establish use of that technology.
    tokens.update(item[:-6] for item in tuple(tokens) if item.endswith("-based") and len(item) > 6)
    return {_CANONICAL.get(item, item) for item in tokens if item and item not in _STOP}


def _nli_premise(quote: str) -> tuple[str, bool]:
    """Resolve the implicit subject only in narrowly recognized CV frames.

    No claims are added to the saved evidence: callers retain the exact original
    quote. Only 'Contributed to ...' and 'Worked on/with/as/at/in ...' can gain a
    subject. Other past-participle openings may be adjectival and stay raw, as
    do multi-sentence windows, imperatives and ambiguous actors.
    This input-format rule is not a calibrated improvement in hiring accuracy.
    """
    safe = (
        len(_spans(quote)) == 1 and "\n" not in quote and len(quote.split()) >= 3
        and not _NEGATIVE.search(quote) and not _ASPIRATIONAL.search(quote)
        and not _OTHER_ACTOR.search(quote) and not _NAMED_OTHER_ACTION.search(quote)
        and not _AMBIGUOUS_ACHIEVEMENT.search(quote)
        # An agent introduced with 'by' may be someone other than the candidate.
        # Numeric outcomes such as 'reduced time by 10%' are unaffected.
        and not re.search(r"\bby\s+[a-z]", quote, re.I)
    )
    if not safe:
        return quote, False
    if _FIRST_PERSON_ACTION.search(quote):
        return quote, True
    if _SUBJECTLESS_FRAME.search(quote):
        return "The candidate " + quote[0].lower() + quote[1:], True
    return quote, False


def _explicit_exemption(text: str) -> bool:
    return bool(re.search(r"\b(?:not required|not essential|no .{0,100} required|no .{0,100} necessary)\b", text, re.I)) and not re.search(
        r"\b(and|or|but|if|unless|except|when|where|provided|conditional|subject to)\b|[,;?]", text, re.I,
    )


def _form_tail_start(text: str) -> int | None:
    """Recognize a narrow, corroborated UI tail; never infer from 'apply' alone.

    Exclusion is always accompanied by assessment coverage uncertainty. The
    source job stays untouched so a person can check the boundary, including
    any employer text that may occur among the form controls.
    """
    for marker in re.finditer(r"^[ \t]*Create a Job Alert[ \t]*\r?$", text, re.I | re.M):
        tail = text[marker.end():marker.end() + 2_000]
        signals = (
            r"^[ \t*]*indicates a required field[ \t]*\r?$",
            r"^[ \t]*Accepted file types:[^\n\r]+$",
            r"^[ \t]*Click here to view [^\n\r]*Privacy Notice[^\n\r]*$",
            r"^[ \t]*Powered by(?:[ \t]+[^\n\r]*)?[ \t]*\r?$",
        )
        if sum(bool(re.search(signal, tail, re.I | re.M)) for signal in signals) >= 2:
            return marker.start()
    return None


def _unheaded_requirement(body: str) -> bool:
    """Do not mistake an employer's 'strong returns' or staff experience for a qualification.

    Unheaded direct obligations remain eligible, including those in a narrative
    section. Unknown headings/lists retain their independent coverage guard.
    """
    if _EXPLICIT_CORE.search(body) or _ELIGIBILITY.search(body):
        return True
    candidate = re.search(r"\b(?:you|your|candidates?|applicants?)\b", body, re.I)
    background = re.search(
        r"^(?:we\b|our\b|the (?:foundation|firm|company|team|department|business)\b|"
        r"speciali[sz]ing\b|as one of\b)", body, re.I,
    )
    if background and not candidate:
        return False
    return bool(_MARKER.search(body) or _DUTY.search(body))


def _preference_scope(body: str) -> tuple[bool, bool]:
    """Return whole-clause preference and unresolved mixed/local scope.

    In particular, a parenthesized language preference cannot make the rest of
    a required stack optional. Keep unresolved mixed scope for manual review.
    """
    outside = re.sub(r"\([^()]*\)", "", body)
    local = bool(_PREFERRED.search(body)) and not _PREFERRED.search(outside)
    preference = _PREFERRED.search(outside)
    # A preference in the middle of coordinated obligations has unclear scope.
    # Recognize whole-clause suffixes and explicit prefix labels, not merely the
    # presence of a keyword anywhere in a requirements paragraph.
    whole = bool(re.search(
        r"(?:preferred|desirable|advantageous|a bonus|(?:a\s+)?(?:great\s+|strong\s+)?plus|"
        r"nice[- ]to[- ]have|would be useful|would be beneficial)[.!;:]?\s*$", outside, re.I,
    ) or re.match(r"^(?:preferred|desirable|nice[- ]to[- ]have)\s*:", outside, re.I))
    mixed = bool(preference and (not whole or
        _EXPLICIT_CORE.search(outside)
        or re.search(r"\b(?:but|whereas|while|as well as)\b", outside, re.I)
    ))
    return bool(preference) and not mixed, local or mixed


def _requirements(text: str) -> list[_Unit]:
    form_start = _form_tail_start(text)
    if form_start is not None:
        text = text[:form_start]
    section = None
    excluded_section = False
    unknown_section = False
    result: dict[str, _Unit] = {}
    rank = {"NOT_REQUIREMENT": 0, "CONTEXTUAL": 0, "PREFERRED": 1, "IMPORTANT": 2, "CORE": 3}
    for raw, start, end in _spans(text):
        body = raw
        prefix = text[text.rfind("\n", 0, start) + 1:start]
        is_bullet = bool(re.fullmatch(r"\s*(?:[-*•]|\d+[.)])\s*", prefix))
        heading = raw.strip(" #:.-").casefold()
        if heading in _HEADINGS:
            section = _HEADINGS[heading]
            excluded_section = section is None and heading not in _NARRATIVE_HEADINGS
            unknown_section = False
            continue
        # An unknown heading must not hide the bare obligations that follow it.
        # Keep the heading itself as unresolved coverage, rather than claiming
        # that recognizing a few explicit requirements covered the whole job.
        possible_heading = not is_bullet and len(raw) <= 120 and (
            raw.endswith(":") or raw.startswith("#") or (
                section is None and not excluded_section and len(raw.split()) <= 10
                and not re.search(r"[.!?;]$", raw) and not _MARKER.search(raw)
            )
        )
        if possible_heading:
            section, excluded_section, unknown_section = "IMPORTANT", False, True
            result["unparsed heading: " + raw.casefold()] = _Unit(
                raw, raw, "IMPORTANT", "An unrecognized section heading requires manual coverage review."
            )
            continue
        if ":" in raw:
            prefix, remainder = raw.split(":", 1)
            if prefix.strip().casefold() in _HEADINGS:
                section = _HEADINGS[prefix.strip().casefold()]
                excluded_section = section is None and prefix.strip().casefold() not in _NARRATIVE_HEADINGS
                unknown_section = False
                body = remainder.strip()
        explicit = _unheaded_requirement(body)
        if not body or (excluded_section and not _EXPLICIT_CORE.search(body) and not _ELIGIBILITY.search(body)):
            continue
        if not (section or explicit or is_bullet):
            continue
        # Reuse existing explicit wording, but never let familiarity disappear
        # into CONTEXTUAL or a preference header become a scored requirement.
        # Reuse the importance baseline only outside scoped parentheticals.
        baseline = classify_importance({"context": re.sub(r"\([^()]*\)", "", body)})["importance_category"]
        preferred, mixed_preference = _preference_scope(body)
        ambiguous = (
            "The obligation is in an unrecognized section and needs manual coverage review."
            if unknown_section else "An unclassified list item needs manual coverage review."
            if is_bullet and section is None else None
        )
        if _explicit_exemption(body):
            importance = "NOT_REQUIREMENT"
        elif _NEGATIVE.search(body):
            importance = "CORE" if _EXPLICIT_CORE.search(body) or section == "CORE" else "IMPORTANT"
            ambiguous = "A negated or conditional requirement needs manual interpretation."
        elif mixed_preference:
            importance = "CORE" if _EXPLICIT_CORE.search(body) or section == "CORE" else (section or "IMPORTANT")
            ambiguous = "Required and preferred wording has local or mixed scope; the complete obligation needs review."
        elif preferred or baseline == "preferred":
            importance = "PREFERRED"
        elif _EXPLICIT_CORE.search(body):
            importance = "CORE"
        elif section:
            importance = section
        else:
            importance = "IMPORTANT"
        key = " ".join(body.casefold().split())
        unit = _Unit(body, raw, importance, ambiguous)
        if key not in result or rank[importance] > rank[result[key].importance]:
            result[key] = unit
    return list(result.values())


def _claim(text: str) -> str:
    body = text.strip().rstrip(".;:")
    body = re.sub(r"^(?:you (?:must|should|will|need to)|candidates? (?:must|should)|must|should)\s+", "", body, flags=re.I)
    body = re.sub(r"^(?:have|possess|demonstrate)\s+", "", body, flags=re.I)
    body = re.sub(r"\s+(?:(?:is|are|would be|will be)\s+)?(?:required|essential|mandatory|preferred|desirable|advantageous|a bonus)$", "", body, flags=re.I)
    return body.strip()


def _atoms(unit: _Unit) -> tuple[list[str], str, str | None]:
    claim = _claim(unit.text)
    if unit.ambiguous:
        return [], "all", unit.ambiguous
    if _ELIGIBILITY.search(claim):
        return [], "all", "Eligibility, citizenship, clearance, licences, or language fluency require explicitly verified facts; CV location and skill mentions are not proof."
    if _QUANTITY.search(claim):
        return [], "all", "Quantities or experience duration require verified interpretation; NLI does not establish years from mentions or dates."
    if _COMPLEX.search(claim):
        return [], "all", "Conditional or equivalent qualifications need manual interpretation."
    if _DEGREE.search(claim):
        return [], "all", "Degree level, subject and completion qualifiers need verified credential evidence; unsupported credential conditions require review."
    shared = decompose_shared_skills(claim)
    if shared is not None:
        return list(shared.atoms), shared.combination, shared.uncertainty
    has_or = bool(re.search(r"\bor\b", claim, re.I))
    has_and = bool(re.search(r"\band\b|[,/&]", claim, re.I))
    if has_or and has_and:
        return [], "all", "Mixed alternatives and conjunctions need manual interpretation."
    if has_or or has_and:
        if _DEGREE.search(claim) or _EXPERIENCE.search(claim) or re.search(r"\b(knowledge|familiarity|understanding|ability|proficien\w*)\b", claim, re.I):
            return [], "all", "Shared qualifications or experience across alternatives/conjunctions need manual interpretation; qualifiers cannot be dropped."
        pieces = [part.strip() for part in re.split(r"\bor\b" if has_or else r"\band\b|[,/&]", claim, flags=re.I) if part.strip()]
        # Multiword pieces can carry a qualifier shared with later alternatives
        # ('advanced Python or SQL', 'maintain Python and SQL services'). Only
        # bare technology tokens use this fallback; qualified lists must use
        # the full-prefix grammar above.
        if len(pieces) < 2 or len(pieces) > 4 or any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9+#.-]*", piece) for piece in pieces
        ):
            return [], "all", "A complex conjunction needs separate verified evidence for all obligations."
        return pieces, "any" if has_or else "all", None
    return [claim], "all", None


def _hypothesis(atom: str) -> str:
    if _DUTY.search(atom):
        return "The candidate can " + re.sub(r"^(?:you will |you'll |to )", "", atom, flags=re.I).lower() + "."
    if re.match(r"ability to\b", atom, re.I):
        return "The candidate is able to " + atom[len("ability to"):].strip() + "."
    if _EXPERIENCE.search(atom) or _DEGREE.search(atom) or re.search(r"\b(knowledge|familiarity|understanding|skills?)\b", atom, re.I):
        return "The candidate has " + atom + "."
    return "The candidate has " + atom + " skills."


def _valid_probabilities(result: dict) -> dict[str, float]:
    try:
        probabilities = result["probabilities"]
        if set(probabilities) != {"entailment", "neutral", "contradiction"}:
            raise ValueError
        values = {name: float(value) for name, value in probabilities.items()}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()) or abs(sum(values.values()) - 1) > 0.02:
            raise ValueError
        predicted = max(values, key=values.get)
        if result["label"] != predicted or abs(float(result["confidence"]) - values[predicted]) > 0.001:
            raise ValueError
        return values
    except (KeyError, TypeError, ValueError):
        raise ModelProviderError("The NLI classifier returned invalid/ambiguous labels or probabilities; matching was stopped.") from None


class ClassifierMatcher:
    name = "classifier"

    def __init__(self, *, classifier=None):
        self.classifier = classifier or RequirementNLIClassifier(local_files_only=True)
        self.diagnostics: dict = {}

    @property
    def fingerprint(self) -> dict:
        configuration = getattr(self.classifier, "fingerprint", None)
        return {
            "matcher": "classifier_matching_v8", "classifier": configuration or {"injected": type(self.classifier).__name__},
            "entailment_threshold": ENTAILMENT_THRESHOLD, "max_requirements": MAX_REQUIREMENTS,
            "max_cv_units": MAX_CV_UNITS, "max_candidates_per_atom": MAX_CANDIDATES,
            "max_pairs": MAX_PAIRS, "max_window_characters": MAX_WINDOW_CHARS,
        }

    def assess(self, job: dict, profile: CandidateProfile, master_cv: str) -> MatchAssessment:
        from app.applications.classifier_constraints import check_constraints
        from app.applications.classifier_education import match_degree

        text = job.get("description")
        if not isinstance(text, str) or not text.strip() or not isinstance(master_cv, str) or not master_cv.strip():
            raise ModelProviderError("Classifier matching needs a nonempty full job description and master CV.")
        self.diagnostics = {
            "baseline": "Existing pretrained general NLI with conservative rules; not custom-trained or calibrated for hiring.",
            "configuration": self.fingerprint, "nli_pairs": 0, "overlong_pairs": 0,
            "normalized_subject_pairs": 0,
            "degree_rule_matches": 0,
            "no_generative_matching": True,
        }
        checks = check_constraints(job, profile)
        notes = []
        complete = len(text.strip()) >= 300 and (job.get("metadata") or {}).get("description_type") != "summary"
        if not complete:
            notes.append("The source description is a summary or too short to establish complete requirements.")
        if len(text) > 80_000 or len(master_cv) > 100_000:
            notes.append("Input exceeds the classifier's bounded source size; no source content was truncated or inferred.")
            units = []
        else:
            units = _requirements(text)
            form_start = _form_tail_start(text)
            if form_start is not None:
                notes.append(
                    "A corroborated application-form UI tail was excluded from scored requirements. "
                    "Review the extraction boundary against the unchanged source to confirm no role content was omitted."
                )
                self.diagnostics["excluded_form_tail"] = {
                    "marker": "Create a Job Alert", "source_start": form_start,
                    "characters": len(text) - form_start, "coverage_review_required": True,
                }
        if len(units) > MAX_REQUIREMENTS:
            notes.append(f"The job contains more than {MAX_REQUIREMENTS} requirement units; complete assessment requires review.")
            units = []
        if not any(unit.importance in {"CORE", "IMPORTANT", "PREFERRED"} for unit in units):
            notes.append("No complete set of scored requirements could be established from explicit wording/sections.")
        spans = _spans(master_cv)
        windows = []
        self._nonprofessional_evidence = set()
        nonprofessional_section = False
        if len(spans) > MAX_CV_UNITS:
            notes.append(f"The CV exceeds {MAX_CV_UNITS} evidence units; the remaining evidence was not searched and requires review.")
        for index, (value, start, end) in enumerate(spans[:MAX_CV_UNITS]):
            if _NONPROFESSIONAL.search(value) and not _ACTION.search(value):
                nonprofessional_section = True
            elif re.fullmatch(r"(?:professional |work )?experience[: ]*|employment(?: history)?[: ]*", value, re.I):
                nonprofessional_section = False
            if nonprofessional_section or _NONPROFESSIONAL.search(value):
                self._nonprofessional_evidence.add(value)
            if len(value) > MAX_WINDOW_CHARS:
                notes.append("A CV evidence unit exceeds the sentence-window allowance and requires review; it was not truncated.")
                continue
            windows.append(value)
            if index + 1 < min(len(spans), MAX_CV_UNITS):
                combined = master_cv[start:spans[index + 1][2]]
                if len(combined) <= MAX_WINDOW_CHARS:
                    windows.append(combined)
        windows = list(dict.fromkeys(windows))
        matches = []
        for unit in units:
            if unit.importance == "NOT_REQUIREMENT":
                matches.append(RequirementMatch(requirement=unit.text, importance="NOT_REQUIREMENT", job_evidence=unit.quote,
                                                status="UNKNOWN", cv_evidence=[], explanation="The advertisement explicitly waives this requirement."))
                continue
            degree = match_degree(unit.text, master_cv) if not unit.ambiguous else None
            if degree is not None:
                status, evidence, explanation = degree
                self.diagnostics["degree_rule_matches"] += int(status == "MET")
                matches.append(RequirementMatch(requirement=unit.text, importance=unit.importance,
                                                job_evidence=unit.quote, status=status,
                                                cv_evidence=evidence, explanation=explanation))
                if status != "MET" and unit.importance != "PREFERRED":
                    notes.append(explanation + " Requirement: " + unit.text)
                continue
            atoms, combination, uncertainty = _atoms(unit)
            coverage_uncertainty = uncertainty
            evidence = []
            verified = []
            if atoms:
                for atom in atoms:
                    found, reason = self._match_atom(atom, unit, windows)
                    verified.append(bool(found))
                    evidence.extend(found)
                    if reason:
                        uncertainty = "; ".join(dict.fromkeys(value for value in (uncertainty, reason) if value))
                    if combination == "any" and found:
                        uncertainty = coverage_uncertainty
                        break
                supported = not uncertainty and bool(verified) and (any(verified) if combination == "any" else all(verified))
            else:
                supported = False
            if supported:
                status, explanation = "MET", "Explicit source evidence passed the lexical/context checks and pretrained NLI entailment threshold."
            elif any(verified):
                status = "PARTIAL"
                explanation = (
                    f"Verified {sum(verified)} of {len(atoms)} explicit components with their original qualifiers. "
                    "The full requirement is not established."
                    + (" " + uncertainty if uncertainty else " Remaining components have no verified evidence.")
                )
                if unit.importance != "PREFERRED" or uncertainty:
                    notes.append(explanation + " Requirement: " + unit.text)
            else:
                status = "MISSING" if unit.importance == "PREFERRED" and not uncertainty else "UNKNOWN"
                explanation = uncertainty or "No complete verified supporting CV evidence was established; a mention alone does not prove the required experience."
                evidence = []
                if unit.importance != "PREFERRED" or uncertainty:
                    notes.append(explanation + " Requirement: " + unit.text)
            matches.append(RequirementMatch(requirement=unit.text, importance=unit.importance, job_evidence=unit.quote,
                                            status=status, cv_evidence=list(dict.fromkeys(evidence)), explanation=explanation))
        if self.diagnostics["overlong_pairs"]:
            notes.append("Some complete NLI evidence pairs exceeded the token allowance; no pair was silently truncated.")
        if self.diagnostics.get("pair_budget_exhausted"):
            notes.append("The bounded NLI pair budget was exhausted; remaining obligations need review.")
        self.diagnostics.update(requirement_units=len(units), cv_units=len(spans), cv_windows=len(windows))
        notes = list(dict.fromkeys(notes))
        review = bool(notes) or any(check.status == "UNKNOWN" for check in checks)
        return MatchAssessment(
            description_complete=complete, requirements=matches, constraint_checks=checks,
            recommended_verdict="REVIEW" if review else "APPLY",
            rationale="Classifier evidence is supplied to the deterministic scorer; this field is not an NLI hiring verdict.",
            uncertainties=notes,
        )

    def _match_atom(self, atom: str, unit: _Unit, windows: list[str]) -> tuple[list[str], str | None]:
        anchors = _tokens(atom)
        if not anchors:
            return [], "The requirement could not be reduced to a meaningful evidence hypothesis."
        candidates = []
        needs_experience = bool(_EXPERIENCE.search(unit.text) or _DUTY.search(unit.text))
        needs_professional = bool(_PROFESSIONAL.search(unit.text))
        for window in windows:
            if _NEGATIVE.search(window) or _ASPIRATIONAL.search(window) or _OTHER_ACTOR.search(window) or _NAMED_OTHER_ACTION.search(window):
                continue
            coverage = len(anchors & _tokens(window)) / len(anchors)
            if coverage < 0.6:
                continue
            if needs_experience and not any(
                _ACTION.search(sentence) and len(anchors & _tokens(sentence)) / len(anchors) >= 0.6
                for sentence, _, _ in _spans(window)
            ):
                continue
            if needs_professional and (
                _NONPROFESSIONAL.search(window)
                or any(quote in window for quote in self._nonprofessional_evidence)
                or not any(
                    _PROVENANCE.search(sentence) and _ACTION.search(sentence)
                    and len(anchors & _tokens(sentence)) / len(anchors) >= 0.6
                    for sentence, _, _ in _spans(window)
                )
            ):
                continue
            if _DEGREE.search(atom) and not _DEGREE.search(window):
                continue
            premise, coherent = _nli_premise(window)
            candidates.append((coherent, coverage, -len(window), window, premise))
        candidates.sort(reverse=True)
        # Prefer distinct evidence before windows that merely surround an
        # already-ranked quote. Deferred windows remain eligible within the cap.
        distinct, overlapping = [], []
        for candidate in candidates:
            quote = candidate[3]
            target = overlapping if any(quote in prior[3] or prior[3] in quote for prior in distinct) else distinct
            target.append(candidate)
        candidates = distinct + overlapping
        for _, _, _, evidence, premise in candidates[:MAX_CANDIDATES]:
            if self.diagnostics["nli_pairs"] >= MAX_PAIRS:
                self.diagnostics["pair_budget_exhausted"] = True
                return [], "The NLI pair budget was exhausted before this requirement was verified."
            self.diagnostics["nli_pairs"] += 1
            self.diagnostics["normalized_subject_pairs"] += int(premise != evidence)
            try:
                result = self.classifier.classify(context=premise, hypothesis=_hypothesis(atom))
            except NLIInputTooLong:
                self.diagnostics["overlong_pairs"] += 1
                continue
            probabilities = _valid_probabilities(result)
            if probabilities["entailment"] >= ENTAILMENT_THRESHOLD and probabilities["contradiction"] <= 0.05:
                return [evidence], None
        if len(candidates) > MAX_CANDIDATES:
            return [], "Only a bounded set of candidate evidence windows could be checked; support remains unresolved."
        return [], None
