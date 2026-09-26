"""Narrow, deterministic matching of completed educational qualifications.

This is a source-evidence rule, not credential verification. It deliberately
does not reason about equivalence, grades, accreditation, duration, experience
substitutions, or arbitrary degree subjects. Unsupported requirement grammar
returns ``None`` so the caller can retain its manual-review safeguards.
"""

from __future__ import annotations

from datetime import date
import re


_SUBJECTS = {
    "computer science": "computer_science", "cs": "computer_science",
    "engineering": "engineering", "operations research": "operations_research",
    "operational research": "operations_research", "mathematics": "mathematics",
    "maths": "mathematics", "math": "mathematics", "statistics": "statistics",
    "physics": "physics", "big data science": "data_science", "data science": "data_science",
    "information technology": "information_technology",
}
_QUANTITATIVE = {
    "computer_science", "engineering", "operations_research", "mathematics",
    "statistics", "physics", "data_science",
}
_LEVELS = {
    "undergraduate": 1, "bachelor": 1, "bachelor's": 1, "bachelors": 1,
    "bsc": 1, "beng": 1, "ba": 1, "bs": 1,
    "master": 2, "master's": 2, "masters": 2, "msc": 2, "meng": 2,
    "phd": 3, "doctorate": 3, "doctoral": 3,
}
_LEVEL_NAMES = "|".join(re.escape(name) for name in sorted(_LEVELS, key=len, reverse=True))
_CREDENTIAL = re.compile(
    r"^(?P<name>bachelor of engineering|bachelor of science|bachelor of arts|"
    r"master of engineering|master of science|master of arts|doctor of philosophy|"
    r"bachelor(?:'s|s)?(?: degree)?|master(?:'s|s)?(?: degree)?|"
    r"bsc|beng|ba|bs|msc|meng|phd|doctorate)\b(?P<subject>.*)$"
)
_COMPLETED_FRAME = re.compile(
    r"^(?:(?:i (?:have )?)?(?:earned|obtained|completed|was awarded|graduated with)|i hold)\s+(?:an?\s+)?"
)
_UNSAFE = re.compile(
    r"\?|\b(?:no|not|never|without|lack|lacking|incomplete|uncompleted|unfinished|"
    r"failed|withdrawn|withdrew|abandoned|dropout|dropped out|pending|deferred|"
    r"pursuing|studying|student|students|enrolled|enrolment|enrollment|"
    r"in progress|ongoing|present|expected|anticipated|prospective|future|"
    r"planning|planned|intend|intended|hope|hoping|want|seeking|aim|aiming|"
    r"course|courses|coursework|module|modules|bootcamp|certificate|certification|"
    r"equivalent|honorary|applicant|applicants|application|admission|conditional|"
    r"brother|sister|wife|husband|friend|colleague|colleagues|mentor|supervisor|"
    r"he|she|they|their|his|her|someone|sample|example|template|fictional|"
    r"could|would|should|may|might|will|if)\b|\b\w+n['’]t\b",
    re.I,
)
_EDUCATION_HEADING = re.compile(r"^(?:education|academic qualifications|educational qualifications|qualifications)$")
_OTHER_HEADING = re.compile(
    r"^(?:(?:professional|work|employment|research|volunteer) )?(?:experience|history)|"
    r"(?:technical |professional )?skills|(?:selected |personal |academic )?projects|"
    r"(?:professional )?summary|employment|references|publications|interests|"
    r"awards(?: and recognition)?|certifications|training$"
)
_DATE_RANGE = re.compile(r"\b(?:19|20)\d{2}\s*[-–—]\s*((?:19|20)\d{2})\b")
_YEAR = re.compile(r"\b((?:19|20|21|29)\d{2})\b")


def _clean(value: str) -> str:
    value = value.replace("’", "'").replace("–", "-").replace("—", "-")
    return " ".join(value.strip().lstrip("#*•- ").strip("* ").casefold().split())


def _other_heading(raw: str) -> bool:
    clean = _clean(raw).rstrip(":")
    return bool(_OTHER_HEADING.fullmatch(clean)) or raw.strip().startswith("#") or (
        raw.strip().isupper() and len(clean.split()) <= 8
    )


def _parse_requirement(requirement: str) -> tuple[set[int], set[str]] | None:
    body = _clean(requirement).rstrip(".;:")
    body = re.sub(
        r"^(?:(?:you|candidates?) (?:must|should|need to) (?:have|hold|possess)|"
        r"(?:must|should) (?:have|hold|possess)|have|hold|possess)\s+", "", body,
    )
    body = re.sub(r"\s+(?:(?:is|are)\s+)?(?:required|essential|mandatory|preferred|desirable)$", "", body)
    body = re.sub(r"^an?\s+", "", body)
    pattern = (
        rf"(?P<levels>(?:{_LEVEL_NAMES})(?:\s+or\s+(?:{_LEVEL_NAMES}))?)"
        r"(?P<higher_before> or (?:higher|above))?(?: degree)?"
        r"(?P<higher_after> or (?:higher|above))? in (?P<subjects>.+)"
    )
    match = re.fullmatch(pattern, body)
    if match:
        levels = {_LEVELS[level] for level in match["levels"].split(" or ")}
        if match["higher_before"] or match["higher_after"]:
            # Explicit 'or higher' applies to a single minimum level only.
            if len(levels) != 1:
                return None
            levels = set(range(min(levels), 4))
        subjects = match["subjects"]
    else:
        match = re.fullmatch(r"degree in (?P<subjects>.+)", body)
        if not match:
            return None
        levels, subjects = {1, 2, 3}, match["subjects"]
    # Commas are accepted only as an explicitly disjunctive list. Nothing
    # silently converts an AND, slash, or an unqualified comma list to OR.
    if re.search(r"\band\b|[/&]", subjects) or ("," in subjects and not re.search(r"\bor\b", subjects)):
        return None
    parts = re.split(r",\s*(?:or\s+)?|\s+or\s+", subjects)
    if not 1 <= len(parts) <= 8 or any(not part for part in parts):
        return None
    alternatives = set()
    for part in parts:
        if part in _SUBJECTS:
            alternatives.add(_SUBJECTS[part])
        elif re.fullmatch(r"(?:(?:another|other|a|any) )?quantitative (?:discipline|field|subject)", part):
            alternatives.add("quantitative")
        else:
            return None
    return levels, alternatives


def _credential(line: str) -> tuple[int, set[str], bool] | None:
    body = _clean(line).rstrip(".")
    explicit = bool(_COMPLETED_FRAME.match(body))
    body = _COMPLETED_FRAME.sub("", body)
    match = _CREDENTIAL.fullmatch(body)
    if not match or _UNSAFE.search(body):
        return None
    name = match["name"]
    if name.startswith("bachelor") or name in {"bsc", "beng", "ba", "bs"}:
        level = 1
    elif name.startswith("master") or name in {"msc", "meng"}:
        level = 2
    else:
        level = 3
    subjects = {"engineering"} if name in {"beng", "meng", "bachelor of engineering", "master of engineering"} else set()
    subject = re.sub(r"^(?:\s*[-:|]\s*|\s+in\s+|\s+)", "", match["subject"])
    for alias in sorted(_SUBJECTS, key=len, reverse=True):
        if re.match(re.escape(alias) + r"(?:\b|$)", subject):
            remainder = subject[len(alias):]
            if re.match(r"\s*(?:and\b|or\b|[/&])", remainder):
                return None
            # The beginning of a different subject (e.g. 'Computer Science
            # Fiction') is not the named discipline. Only familiar credential
            # boundaries/metadata may follow an exact subject name.
            if remainder and not re.match(
                r"^(?:\s*$|\s*[.|,;:\-\d]|\s+(?:with (?:distinction|merit|honou?rs)\b|from\b|at\b))",
                remainder,
            ):
                return None
            subjects.add(_SUBJECTS[alias])
            break
    return level, subjects, explicit


def _evidence(cv: str) -> list[tuple[int, set[str], str]]:
    """Read bounded credential records and preserve their exact source slices."""
    lines = list(re.finditer(r"[^\n\r]+", cv))
    records = []
    in_education = False
    current_year = date.today().year
    for index, line in enumerate(lines):
        raw, clean = line.group(), _clean(line.group())
        if _EDUCATION_HEADING.fullmatch(clean.rstrip(":")):
            in_education = True
            continue
        parsed = _credential(raw)
        if parsed is None:
            # Recognize headings only for section membership, not evidence.
            if _other_heading(raw):
                in_education = False
            continue
        level, subjects, explicit = parsed
        end = line.end()
        if in_education:
            for following in lines[index + 1:index + 4]:
                following_raw = following.group()
                if _credential(following_raw) or _other_heading(following_raw):
                    break
                if following.end() - line.start() > 640:
                    break
                end = following.end()
        quote = cv[line.start():end]
        if len(quote) > 640 or _UNSAFE.search(quote):
            continue
        years = [int(value) for value in _YEAR.findall(quote)]
        if years and max(years) > current_year:
            continue
        # A degree title alone can describe a current enrolment. Bare entries
        # need a past academic date range or an explicit completed/awarded date.
        past_range = any(int(value) < current_year for value in _DATE_RANGE.findall(quote))
        dated_completion = bool(re.search(r"\b(?:completed|awarded|graduated)\b", quote, re.I)) and bool(years) and max(years) < current_year
        if explicit or past_range or (in_education and dated_completion):
            records.append((level, subjects, quote))
    return records


def match_degree(requirement: str, cv: str) -> tuple[str, list[str], str] | None:
    """Return MET/UNKNOWN with exact CV quotes for a supported degree clause.

    Subject alternatives retain their shared degree level, and a postgraduate
    title never supplies a missing subject. Unknown completion, unsupported
    subjects, and absent evidence remain UNKNOWN, never MISSING.
    """
    parsed = _parse_requirement(requirement)
    if parsed is None:
        return None
    levels, alternatives = parsed
    if len(cv) <= 100_000:
        for level, subjects, quote in _evidence(cv):
            matches_subject = bool(subjects & alternatives) or (
                "quantitative" in alternatives and bool(subjects & _QUANTITATIVE)
            )
            if level in levels and matches_subject:
                return "MET", [quote], "A completed degree entry explicitly supports an allowed degree level and subject alternative; the original CV quote is retained."
    return "UNKNOWN", [], "The shared degree-level and subject qualifiers were preserved, but no completed matching credential was established from explicit CV evidence."
