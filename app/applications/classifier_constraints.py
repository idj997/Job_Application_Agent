"""Conservative, evidence-backed checks for structured candidate preferences.

These rules interpret a deliberately small set of explicit statements. They do
not guess work authorization, read headquarters as a job location, or attempt to
interpret arbitrary natural-language constraints.
"""

from __future__ import annotations

import re

from app.applications.models import CandidateProfile, ConstraintCheck


_COUNTRIES = {
    "uk": ("united kingdom", "uk", "u.k.", "great britain", "britain", "england", "scotland", "wales", "northern ireland"),
    "india": ("india",),
    "usa": ("united states", "united states of america", "usa", "u.s.a."),
    "canada": ("canada",), "germany": ("germany",), "france": ("france",),
    "ireland": ("republic of ireland", "ireland"), "netherlands": ("netherlands",),
    "switzerland": ("switzerland",), "spain": ("spain",), "italy": ("italy",),
    "poland": ("poland",), "australia": ("australia",), "singapore": ("singapore",),
    "china": ("china",), "japan": ("japan",), "taiwan": ("taiwan",),
}
_CONDITIONAL = re.compile(r"\b(?:may|might|could|potentially|possibly|conditional|depending|subject to|case[- ]by[- ]case|where possible|eligible|if|provided that|as long as)\b", re.I)
_NEGATIVE = re.compile(r"\b(?:not|no|never|cannot|can't|unable|without|won't|don't|doesn't)\b", re.I)
_LOCATION = re.compile(
    r"^(?:(?:job|work|role|office)\s+)?locations?\s*:|"
    r"\b(?:this|the|your)\s+(?:role|job|position)\s+(?:is|will (?:not )?be|may (?:not )?be|might (?:not )?be|could (?:not )?be)\s+(?:not\s+)?(?:based|located)\s+(?:in|at)\b|"
    r"\b(?:this|the)\s+(?:role|job|position)\s+(?:is|may be|might be|could be)\s+(?:not\s+)?in\b|"
    r"\byou\s+(?:will|must)\s+be\s+(?:based|located)\s+in\b|"
    r"\b(?:must(?: not)?|cannot|can not|can't|can|may|might|could)\s+work\s+(?:from|in|at)\b|"
    r"\b(?:this|the|your)\s+(?:role|job|position)\s+(?:requires?|may require|might require|could require)\s+(?:working|work)\s+(?:from|in|at)\b|"
    r"\bremote\s+(?:work(?:ing)?\s+)?(?:from|within|in)\b",
    re.I,
)
_REQUIRED_COMMUTE = re.compile(
    r"\b(?:(?:this|the|your)\s+(?:role|job|position)|you|candidates?|applicants?)\b.{0,70}\bcommut(?:e|ing)\b|"
    r"\b(?:must|required|requires?|ability|able)\b.{0,35}\bcommut(?:e|ing)\b|"
    r"\bcommut(?:e|ing)\b.{0,100}\b(?:required|mandatory|essential)\b",
    re.I,
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text) if part.strip()]


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase.strip()) + r"(?!\w)", text, re.I))


def _countries(text: str) -> set[str]:
    found = {country for country, aliases in _COUNTRIES.items() if any(_contains(text, alias) for alias in aliases)}
    if _contains(text, "northern ireland") and not _contains(text, "republic of ireland"):
        found.discard("ireland")
    return found


def _quote(text: str, statements: list[str]) -> str | None:
    if not statements:
        return None
    starts = [text.find(statement) for statement in statements]
    if any(start < 0 for start in starts):
        return None
    return text[min(starts):max(start + len(statement) for start, statement in zip(starts, statements))]


def _check(identifier: str, status: str, explanation: str, evidence: str | None = None,
           source: str = "description") -> ConstraintCheck:
    return ConstraintCheck(constraint_id=identifier, status=status, job_evidence=evidence,
                           job_evidence_source=source, explanation=explanation)


def _location_statements(text: str) -> list[str]:
    return [sentence for sentence in _sentences(text)
            if (_LOCATION.search(sentence) or _REQUIRED_COMMUTE.search(sentence))
            and (not re.search(r"\b(?:headquarters|headquartered|customers|clients|offices worldwide)\b", sentence, re.I)
                 or re.search(r"\b(?:this|the|your)\s+(?:role|job|position)\b", sentence, re.I))]


def _ambiguous_location(statement: str) -> bool:
    return bool(_CONDITIONAL.search(statement) or re.search(
        r"\b(?:or|multiple|various|worldwide|globally|relocation options)\b|/", statement, re.I
    ) or len(_countries(statement)) > 1)


def _negative_location(statement: str) -> bool:
    return bool((_REQUIRED_COMMUTE.search(statement) and _NEGATIVE.search(statement)) or re.search(
        r"\bnot\s+(?:be\s+)?(?:based|located|in)\b|\b(?:must not|cannot|can not|can't)\s+work\s+(?:from|in|at)\b|"
        r"\blocations?\s*:\s*not\b|\boutside\s+(?:the\s+)?(?:uk|united kingdom|india)\b",
        statement, re.I,
    ))


def _commute_destination(statement: str) -> str | None:
    """Extract only an explicit commute destination, never infer a city's country."""
    if len(re.findall(r"\boffices?\b", statement, re.I)) > 1:
        return None
    patterns = (
        r"\bcommut(?:e|ing)\s+(?:into|to|in)\s+(?:(?:our|the|an?)\s+)?(?:offices?|workplace)\s+in\s+(.+?)(?=\s+(?:on|each|every|daily|weekly|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|is required)\b|[.;]|$)",
        r"\bcommut(?:e|ing)\s+(?:into|to|in)\s+(?:(?:our|the|an?)\s+)?(.+?)\s+offices?\b",
        r"\bcommut(?:e|ing)\s+(?:into|to|in)\s+(?:(?:our|the|an?)\s+)?(.+?)(?=\s+(?:on|each|every|daily|weekly|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|is required)\b|[.;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, statement, re.I)
        if not match:
            continue
        destination = match.group(1).strip(" ,")
        if (not destination or len(destination) > 80 or _ambiguous_location(destination)
                or _NEGATIVE.search(destination)
                or re.search(r"\b(?:our|the|office|offices|workplace|designated|assigned|nearest|local|chosen|headquarters|hq)\b", destination, re.I)):
            return None
        return destination
    return None


def _destination_agrees(destination: str, location: str) -> bool:
    if _contains(location, destination):
        return True
    # E.g. "London, UK" and "London, England, United Kingdom" agree only
    # because both the exact city text and an explicit country are present.
    parts = [part.strip() for part in destination.split(",") if part.strip()]
    return bool(len(parts) > 1 and _contains(location, parts[0])
                and _countries(destination) and _countries(destination) == _countries(location)
                and all(_countries(part) for part in parts[1:]))


def role_family(title: str) -> str | None:
    """Map only well-scoped target titles to the three application profiles.

    Keep this intentionally conservative. Shared words such as "performance",
    "AI", or "research" are not enough on their own to establish equivalence.
    """
    if re.search(r"\b(?:sales|recruit(?:er|ment|ing)?|marketing|account|product manager|project manager)\b", title, re.I):
        return None

    # Hardware design/verification and GPU-kernel specialties are deliberately
    # not treated as systems-performance equivalents.
    if re.search(r"\b(?:rtl|verification|physical design|logic design|cuda|triton|gpu kernel)\b", title, re.I):
        return None

    if (
        re.search(r"\b(?:cpu|processor|systems?|software|workload)\b", title, re.I)
        and re.search(r"\bperformance\b", title, re.I)
        and re.search(r"\b(?:engineer|engineering|architect|analysis|analyst|tools?)\b", title, re.I)
    ) or re.search(r"\bperformance tools? engineer\b", title, re.I):
        return "cpu_system_performance"

    if re.search(
        r"\b(?:data scientist|data science|decision scientist|research data scientist|applied data scientist)\b",
        title, re.I,
    ):
        return "data_science_applied_ml"

    if (
        re.search(r"\b(?:machine learning|\bml\b|artificial intelligence|\bai\b|\bllm\b)\b", title, re.I)
        and re.search(r"\b(?:engineer|engineering|researcher|scientist)\b", title, re.I)
    ) or re.search(r"\bapplied scientist\b", title, re.I):
        return "ai_ml_research_engineering"

    return None


# Backward-compatible private alias for older imports/tests.
_role_family = role_family

def _normal_role(title: str) -> str:
    title = re.sub(r"\b(?:senior|junior|principal|staff|lead|graduate)\b", "", title, flags=re.I)
    title = re.sub(r"\bml\b", "machine learning", title, flags=re.I)
    title = re.sub(r"\bai\b", "artificial intelligence", title, flags=re.I)
    return " ".join(re.sub(r"[^\w]+", " ", title.casefold()).split())


def _role(job: dict, targets: list[str], text: str) -> ConstraintCheck:
    raw_title = job.get("title")
    title = raw_title.strip() if isinstance(raw_title, str) else ""
    advertised = []
    for statement in _sentences(text):
        label = re.match(r"^(?:job title|role title|title|position|role)\s*:\s*(.+)$", statement, re.I)
        if label:
            advertised.append((label.group(1).strip().rstrip("."), statement))
            continue
        opening = re.search(
            r"\b(?:hir(?:e|ing)|seek(?:ing)?|recruit(?:ing)?|looking for)\s+(?:an?\s+)?(.+)|"
            r"\b(?:this|the)\s+(?:role|job|position)\s+(?:is|may be|might be|could be)\s+(?:an?\s+)?(.+)",
            statement, re.I,
        )
        if opening:
            candidate = next(value for value in opening.groups() if value)
            candidate = re.split(r"\s+(?:to|who|with|for|within|at|in|on)\b|[.;]", candidate, maxsplit=1, flags=re.I)[0].strip()
            if re.search(r"\b(?:engineer|scientist|architect|researcher|manager|recruiter)\b", candidate, re.I):
                advertised.append((candidate, statement))
        elif title and _normal_role(statement.rstrip(".")) == _normal_role(title):
            advertised.append((title, statement))
    if any(_NEGATIVE.search(statement) or _CONDITIONAL.search(statement)
           or re.search(r"\b(?:or|multiple|various)\b", candidate, re.I)
           for candidate, statement in advertised):
        return _check("role", "UNKNOWN", "The advertised job title is conditional, negated or ambiguous.",
                      _quote(text, [statement for _, statement in advertised]))
    if title:
        if _NEGATIVE.search(title) or _CONDITIONAL.search(title) or re.search(r"\b(?:or|multiple|various)\b", title, re.I):
            return _check("role", "UNKNOWN", "The structured job title is conditional, negated or ambiguous.", title, "title")
        if any(_normal_role(candidate) != _normal_role(title) for candidate, _ in advertised):
            return _check("role", "UNKNOWN", "The description and structured job title conflict; review the advertised role.",
                          _quote(text, [statement for _, statement in advertised]))
        evidence, source = title, "title"
    elif advertised and len({_normal_role(candidate) for candidate, _ in advertised}) == 1:
        title, evidence = advertised[0]
        source = "description"
    else:
        return _check("role", "UNKNOWN", "No unambiguous advertised job title was established.")
    if any(_normal_role(title) == _normal_role(target) for target in targets):
        return _check("role", "PASS", "The advertised title matches a configured target role.", evidence, source)
    family = _role_family(title)
    families = [_role_family(target) for target in targets]
    if family and family in families:
        return _check("role", "PASS", "The explicit job title belongs to a configured role family.", evidence, source)
    if family and all(families) and family not in families:
        return _check("role", "FAIL", "The explicit job title belongs to a different role family from the configured targets.", evidence, source)
    return _check("role", "UNKNOWN", "The job title needs review against the configured role targets.", evidence, source)


def _location_result(allowed: list[str], text: str, statements: list[str],
                     source: str = "description") -> ConstraintCheck:
    def broad_country(place: str) -> set[str]:
        value = re.sub(r"^(?:anywhere|any location)\s+(?:in|within)\s+", "", place.strip(), flags=re.I)
        value = re.sub(r"\s*\(?anywhere\)?$", "", value, flags=re.I).strip()
        return {country for country, aliases in _COUNTRIES.items() if value.casefold() in aliases}

    allowed_countries = set().union(*(broad_country(place) for place in allowed))
    results = []
    for statement in statements:
        if (_REQUIRED_COMMUTE.search(statement)
                and (_NEGATIVE.search(statement) or _commute_destination(statement) is None)):
            return _check("location", "UNKNOWN", "The required commute is negated or its destination is unclear.", statement, source)
        if _ambiguous_location(statement):
            return _check("location", "UNKNOWN", "The job has conditional or multiple possible work locations.", statement, source)
        countries = _countries(statement)
        matches = bool(countries & allowed_countries) or any(_contains(statement, place) for place in allowed)
        if _negative_location(statement):
            results.append("FAIL" if matches else "UNKNOWN")
        elif matches:
            results.append("PASS")
        elif countries and allowed_countries and all(broad_country(place) for place in allowed):
            results.append("FAIL")
        else:
            results.append("UNKNOWN")
    if results and set(results) == {"PASS"}:
        return _check("location", "PASS", "The explicit work location matches the configured allowed locations.", _quote(text, statements), source)
    if results and set(results) == {"FAIL"}:
        return _check("location", "FAIL", "The explicit work location excludes the configured allowed locations.", _quote(text, statements), source)
    return _check("location", "UNKNOWN", "A single acceptable work location could not be established.", _quote(text, statements), source)


def _location(allowed: list[str], text: str, job: dict) -> ConstraintCheck:
    statements = _location_statements(text)
    described = _location_result(allowed, text, statements)
    value = job.get("location")
    if not isinstance(value, str) or not value.strip():
        return described
    value = value.strip()
    if (_ambiguous_location(value) or _NEGATIVE.search(value)
            or re.search(r"\b(?:headquarters|headquartered|customers|clients)\b", value, re.I)):
        return _check("location", "UNKNOWN", "The structured location is conditional, ambiguous or not an actual work location.", value, "location")
    structured = _location_result(allowed, value, [value], "location")
    if statements:
        metadata_countries = _countries(value)
        for statement in statements:
            conflict = bool(_ambiguous_location(statement) or _negative_location(statement)
                            or (metadata_countries and _countries(statement) and metadata_countries != _countries(statement)))
            if _REQUIRED_COMMUTE.search(statement):
                destination = _commute_destination(statement)
                agrees = bool(destination and _destination_agrees(destination, value))
            else:
                described_statement = _location_result(allowed, text, [statement])
                agrees = described_statement.status == structured.status and described_statement.status != "UNKNOWN"
            if conflict or not agrees or structured.status == "UNKNOWN":
                return _check("location", "UNKNOWN", "The description and structured work location conflict or need reconciliation.", _quote(text, statements))
    return structured


def _work_pattern(allowed: list[str], text: str) -> ConstraintCheck:
    statements, outcomes = [], []
    for statement in _sentences(text):
        if not re.search(
            r"^(?:working (?:pattern|arrangement)|work (?:pattern|arrangement)|workplace type)\s*:|"
            r"\b(?:this|the)\s+(?:role|job|position)\s+(?:is|will be)\s+(?:(?:not|a|an|fully|entirely|100%)\s+)*(?:remote|hybrid|on[- ]?site|office[- ]based)\b|"
            r"^(?:remote|hybrid|on[- ]?site)\s+work(?:ing)?\s*(?::|is\b)|"
            r"^(?:fully|entirely|100%)\s+remote\b",
            statement, re.I,
        ):
            continue
        patterns = set()
        if re.search(r"\bremote\b", statement, re.I):
            patterns.add("remote")
        if re.search(r"\bhybrid\b", statement, re.I):
            patterns.add("hybrid")
        if re.search(r"\b(?:on[- ]?site|office[- ]based)\b", statement, re.I):
            patterns.add("onsite")
        if not patterns:
            continue
        statements.append(statement)
        if _CONDITIONAL.search(statement) or len(patterns) != 1:
            outcomes.append("UNKNOWN")
        elif re.search(r"\b(?:not|no)\s+(?:(?:a|an|fully)\s+)?(?:remote|hybrid|on[- ]?site|office[- ]based)\b|"
                       r"\b(?:remote|hybrid|on[- ]?site)\s+(?:work(?:ing)?\s+)?(?:is\s+)?(?:not|unavailable)\b", statement, re.I):
            outcomes.append("FAIL" if set(allowed) <= patterns else "UNKNOWN")
        else:
            outcomes.append("PASS" if patterns & set(allowed) else "FAIL")
    if outcomes and len(set(outcomes)) == 1 and outcomes[0] != "UNKNOWN":
        status = outcomes[0]
        return _check("work_pattern", status, "The explicit working arrangement " + ("matches" if status == "PASS" else "does not match") + " the configured preference.", _quote(text, statements))
    return _check("work_pattern", "UNKNOWN", "The working arrangement is missing, conditional or contradictory.", _quote(text, statements))


def _exemption_location(text: str, exemptions: list[str]) -> tuple[bool, bool, str | None]:
    """Return (definite exemption, definite nonexemption, supporting quote)."""
    statements = _location_statements(text)
    if re.search(r"\b(?:fully|entirely|100%)\s+remote\b|\bremote\s+(?:role|position|job|working)\b|"
                 r"\b(?:role|position|job)\s+(?:is|will be)\s+remote\b|"
                 r"\b(?:work(?:ing)? (?:pattern|arrangement)|workplace type)\s*:\s*remote\b", text, re.I):
        return False, False, None
    if not statements or any(_ambiguous_location(value) or _negative_location(value)
                             or re.search(r"\bremote\b", value, re.I) for value in statements):
        return False, False, None
    countries = [_countries(statement) for statement in statements]
    if any(len(value) != 1 for value in countries) or len(set.union(*countries)) != 1:
        return False, False, None
    accepted = set().union(*(_countries(place) for place in exemptions))
    if not accepted or len(accepted) != len(exemptions):
        return False, False, None
    exempt = bool(countries[0] & accepted)
    return exempt, not exempt, _quote(text, statements)


def _sponsorship(text: str, exemptions: list[str], job: dict) -> ConstraintCheck:
    exempt, nonexempt, location_evidence = _exemption_location(text, exemptions) if exemptions else (False, True, None)
    if exemptions and (exempt or nonexempt) and job.get("location"):
        # Metadata cannot create an exemption; contradictory metadata invalidates
        # one otherwise inferred from the description's explicit work location.
        location = _location(exemptions, text, job)
        if location.status == "UNKNOWN" or (location.status == "PASS") != exempt:
            exempt, nonexempt = False, False
    if exempt:
        return _check("sponsorship", "PASS", "The role is explicitly based in a configured sponsorship-exempt country.", location_evidence)
    positive, negative, uncertain = [], [], []
    work_countries = set().union(*(_countries(value) for value in _location_statements(text)))
    structured_location = job.get("location")
    if (isinstance(structured_location, str) and not _ambiguous_location(structured_location)
            and not _NEGATIVE.search(structured_location)
            and not re.search(r"\b(?:headquarters|headquartered)\b", structured_location, re.I)):
        metadata_countries = _countries(structured_location)
        if metadata_countries and (not work_countries or metadata_countries == work_countries):
            # The field can identify a mismatched sponsorship offer, but cannot
            # itself establish any offer or a sponsorship exemption.
            work_countries = metadata_countries
    for statement in _sentences(text):
        if not re.search(r"\b(?:visa|sponsor(?:ship)?|right to work|work authori[sz]ation|work permit)\b", statement, re.I):
            continue
        if _CONDITIONAL.search(statement) or re.search(r"\b(?:but|except|however|only for|for some)\b", statement, re.I):
            uncertain.append(statement)
            continue
        if (re.search(r"\b(?:other|different)\s+(?:roles|positions|vacancies)\b|\b(?:our|the)\s+.{0,30}\boffice\b|"
                      r"\b(?:for|to)\s+(?:candidates|applicants)\s+(?:who|with|holding)\b", statement, re.I)
                or (_countries(statement) and work_countries and not _countries(statement) & work_countries)):
            uncertain.append(statement)
            continue
        denied = bool(re.search(
            r"\bno\s+(?:visa\s+)?sponsorship\b|"
            r"\b(?:cannot|can not|can't|unable to|do not|don't|will not|won't|not able to)\s+(?:currently\s+)?(?:(?:provide|offer|support)\s+)?(?:visa\s+)?sponsor(?:ship)?\b|"
            r"\bsponsorship\s+(?:is\s+)?(?:not\s+(?:available|offered|provided|supported)|unavailable)\b|"
            r"\bmust\s+not\s+require\s+(?:visa\s+)?sponsorship\b|"
            r"\bmust\b.{0,30}\b(?:already|currently|existing)\b.{0,40}\b(?:right to work|work authori[sz]ation)\b|"
            r"\b(?:right to work|work authori[sz]ation)\b.{0,35}\bwithout\s+(?:visa\s+)?sponsorship\b",
            statement, re.I,
        ))
        if denied:
            negative.append(statement)
            continue
        offered = bool(re.search(
            r"\bvisa sponsorship\s*(?::\s*|\s+(?:is\s+)?)(?:available|offered|provided|supported)\b|"
            r"\b(?:we|employer|company)\s+(?:(?:can|will|do)\s+)?(?:offer|provide|support)\s+visa sponsorship\b|"
            r"\b(?:we|employer|company)\s+(?:can|will|do)\s+sponsor\s+(?:(?:skilled worker|work)\s+)?visas\b",
            statement, re.I,
        ))
        if offered and not _NEGATIVE.search(statement):
            positive.append(statement)
        else:
            uncertain.append(statement)
    if uncertain or positive and negative:
        return _check("sponsorship", "UNKNOWN", "Sponsorship terms are conditional, incomplete or contradictory.", _quote(text, positive + negative + uncertain))
    if positive:
        return _check("sponsorship", "PASS", "The employer explicitly offers visa sponsorship for the role.", _quote(text, positive))
    if negative and nonexempt:
        return _check("sponsorship", "FAIL", "The role explicitly excludes required sponsorship or requires existing work authorization.", _quote(text, negative))
    return _check("sponsorship", "UNKNOWN", "Suitable visa sponsorship or an applicable location exemption could not be established.", _quote(text, negative))


_AMOUNT = r"\d+(?:,\d{3})*(?:\.\d+)?\s*[kKmM]?"
_CURRENCY = r"(?:GBP|USD|EUR|INR|US\$|£|€|₹)"
_PAY = re.compile(r"(?P<currency>" + _CURRENCY + r")\s*(?P<low>" + _AMOUNT + r")(?:\s*(?:[-–—]|to)\s*(?:(?P<high_currency>" + _CURRENCY + r")\s*)?(?P<high>" + _AMOUNT + r"))?", re.I)


def _amount(value: str, inherited_suffix: str = "") -> float:
    cleaned = value.replace(",", "").replace(" ", "").lower()
    suffix = cleaned[-1] if cleaned[-1] in "km" else (inherited_suffix if float(cleaned) < 1000 else "")
    number = cleaned[:-1] if cleaned[-1] in "km" else cleaned
    return float(number) * {"": 1, "k": 1000, "m": 1000000}[suffix]


def _salary(job: dict, minimum: int, currency: str, text: str) -> ConstraintCheck:
    if (job.get("metadata") or {}).get("salary_is_predicted") not in (None, False, 0, "0", "false", "False"):
        return _check("salary", "UNKNOWN", "The available salary is marked as predicted.")
    records = []
    for statement in _sentences(text):
        if not re.search(r"\b(?:salary|base pay|base compensation)\b", statement, re.I):
            continue
        if not re.search(r"\b(?:annual(?:ly)?|per annum|per year|a year)\b|\bp\.a\.", statement, re.I):
            continue
        if re.search(r"\b(?:estimated|predicted|approximately|hour|hourly|day|daily|week|weekly|month|monthly|bonus|commission|ote|from|starting|at least|minimum|over|upwards|pro rata|pro[- ]rated|full[- ]time equivalent|fte|part[- ]time)\b", statement, re.I) or _CONDITIONAL.search(statement):
            return _check("salary", "UNKNOWN", "The salary amount, pay period or base-pay component is uncertain.", statement)
        offers = list(_PAY.finditer(statement))
        if len(offers) != 1:
            return _check("salary", "UNKNOWN", "The annual salary range could not be interpreted unambiguously.", statement)
        offer = offers[0]
        unit = {"£": "GBP", "€": "EUR", "₹": "INR", "US$": "USD"}.get(offer["currency"].upper(), offer["currency"].upper())
        if offer["high_currency"]:
            other = {"£": "GBP", "€": "EUR", "₹": "INR", "US$": "USD"}.get(offer["high_currency"].upper(), offer["high_currency"].upper())
            if other != unit:
                return _check("salary", "UNKNOWN", "The salary range mixes currencies.", statement)
        if unit != currency.upper():
            return _check("salary", "UNKNOWN", "The advertised salary uses a different currency; no exchange rate was assumed.", statement)
        high = offer["high"] or offer["low"]
        suffix = high.strip()[-1].lower() if high.strip()[-1].lower() in "km" else ""
        lower, upper = _amount(offer["low"], suffix), _amount(high)
        if lower > upper or _NEGATIVE.search(statement):
            return _check("salary", "UNKNOWN", "The advertised salary range needs manual interpretation.", statement)
        records.append((upper >= minimum, statement))
    if records and len({passed for passed, _ in records}) == 1:
        passed = records[0][0]
        return _check("salary", "PASS" if passed else "FAIL", "The explicit annual salary range " + ("can meet" if passed else "is below") + " the configured minimum; any eventual offer must still be checked.", _quote(text, [statement for _, statement in records]))
    return _check("salary", "UNKNOWN", "A clear annual salary in the configured currency is missing or contradictory.")


def check_constraints(job: dict, profile: CandidateProfile) -> list[ConstraintCheck]:
    """Check each preference once, retaining verbatim, source-tagged job evidence."""
    preferences = profile.preferences
    text = str(job.get("description") or "")
    evaluators = {
        "role": lambda: _role(job, preferences.target_roles, text),
        "location": lambda: _location(preferences.locations, text, job),
        "work_pattern": lambda: _work_pattern(preferences.work_patterns, text),
        "salary": lambda: _salary(job, preferences.minimum_salary, preferences.salary_currency, text),
        "sponsorship": lambda: _sponsorship(text, getattr(preferences, "sponsorship_exempt_locations", []), job),
    }
    return [evaluators[identifier]() if identifier in evaluators else _check(
        identifier, "UNKNOWN", "This custom constraint requires explicit review; arbitrary free text is not interpreted automatically."
    ) for identifier in preferences.constraints()]
