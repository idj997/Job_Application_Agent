import pytest

from app.applications.classifier_constraints import check_constraints
from app.applications.models import CandidateProfile, Preferences


CPU_ROLES = [
    "CPU Performance Engineer", "Processor Performance Engineer",
    "Systems Performance Engineer", "Software Performance Engineer",
    "Performance Tools Engineer", "CPU Performance Architect",
]
AI_ROLES = [
    "AI Engineer", "Machine Learning Engineer", "ML Research Engineer",
    "Research Engineer", "Applied Scientist", "LLM Evaluation Engineer",
]
DATA_ROLES = [
    "Data Scientist", "Senior Data Scientist", "Applied Data Scientist",
    "Decision Scientist", "Research Data Scientist",
]


def checks(description, *, title="", location="", metadata=None, **preferences):
    profile = CandidateProfile(preferences=Preferences(**preferences))
    job = {"title": title, "location": location, "description": description, "metadata": metadata or {}}
    results = check_constraints(job, profile)
    assert [result.constraint_id for result in results] == list(profile.preferences.constraints())
    for result in results:
        if result.status != "UNKNOWN":
            assert result.job_evidence and result.job_evidence in job[result.job_evidence_source]
        elif result.job_evidence:
            assert result.job_evidence in job[result.job_evidence_source]
    return {result.constraint_id: result for result in results}


@pytest.mark.parametrize("title", [
    "CPU Performance Engineer", "Senior Processor Performance Engineer",
    "Systems Performance Engineer", "Software Performance Engineer",
    "Performance Tools Engineer",
])
def test_performance_roles_match_cpu_system_performance_targets(title):
    assert checks(f"Job title: {title}\nProfile workloads and analyze performance.", title=title,
                  target_roles=CPU_ROLES)["role"].status == "PASS"


@pytest.mark.parametrize("title", [
    "AI Engineer", "Senior Machine Learning Engineer", "ML Research Engineer",
    "LLM Evaluation Engineer", "Applied Scientist",
])
def test_ai_ml_roles_match_ai_ml_targets(title):
    assert checks(f"Job title: {title}\nBuild learning systems.", title=title,
                  target_roles=AI_ROLES)["role"].status == "PASS"


@pytest.mark.parametrize("title", [
    "Data Scientist", "Senior Data Scientist", "Applied Data Scientist",
    "Decision Scientist", "Research Data Scientist",
])
def test_data_science_roles_match_data_science_targets(title):
    assert checks(f"Job title: {title}\nAnalyze data and build statistical models.", title=title,
                  target_roles=DATA_ROLES)["role"].status == "PASS"


@pytest.mark.parametrize("title,targets", [
    ("Data Scientist", CPU_ROLES),
    ("Machine Learning Engineer", CPU_ROLES),
    ("CPU Performance Engineer", AI_ROLES),
    ("Machine Learning Engineer", DATA_ROLES),
    ("Data Scientist", AI_ROLES),
])
def test_explicitly_different_refined_role_family_fails(title, targets):
    assert checks(f"Job title: {title}", title=title, target_roles=targets)["role"].status == "FAIL"


@pytest.mark.parametrize("title,targets", [
    ("AI Product Manager", AI_ROLES),
    ("AI Sales Engineer", AI_ROLES),
    ("AI Recruiter", AI_ROLES),
    ("GPU Sales Engineer", CPU_ROLES),
    ("CUDA Engineer", CPU_ROLES),
    ("Distributed Systems Engineer", AI_ROLES),
    ("Performance Engineer", CPU_ROLES),
])
def test_related_words_do_not_establish_a_target_profile(title, targets):
    assert checks(f"Job title: {title}", title=title, target_roles=targets)["role"].status == "UNKNOWN"


def test_explicit_title_field_can_establish_role_with_verifiable_source():
    result = checks(
        "Profile Linux workloads and investigate bottlenecks.",
        title="CPU Performance Engineer",
        target_roles=CPU_ROLES,
    )["role"]
    assert result.status == "PASS"
    assert result.job_evidence_source == "title"
    assert result.job_evidence == "CPU Performance Engineer"


def test_abbreviated_role_description_agrees_with_structured_title():
    result = checks(
        "The company is seeking a ML Engineer to join our team.",
        title="Machine Learning Engineer",
        target_roles=AI_ROLES,
    )["role"]
    assert result.status == "PASS"
    assert result.job_evidence_source == "title"


@pytest.mark.parametrize("description", [
    "Job title: AI Engineer",
    "We are seeking a AI Engineer to join our team.",
    "We are not hiring a Machine Learning Engineer.",
    "We do not hire a Machine Learning Engineer.",
    "We do not seek a Machine Learning Engineer.",
    "We may be hiring a Machine Learning Engineer.",
    "This role is not a Machine Learning Engineer.",
    "Job title: Machine Learning Engineer or Data Scientist",
])
def test_title_field_does_not_override_conflicting_or_uncertain_description(description):
    result = checks(description, title="Machine Learning Engineer", target_roles=AI_ROLES)["role"]
    assert result.status == "UNKNOWN"


@pytest.mark.parametrize("title", [
    "Not a Machine Learning Engineer",
    "Machine Learning Engineer or Data Scientist",
    "Potentially Machine Learning Engineer",
])
def test_ambiguous_structured_title_is_unknown(title):
    assert checks("Build software.", title=title, target_roles=AI_ROLES)["role"].status == "UNKNOWN"


def test_location_field_can_establish_country_without_invented_description_text():
    result = checks("Must commute into the London office Mondays-Fridays.",
                    location="London, England, United Kingdom", locations=["United Kingdom"])["location"]
    assert result.status == "PASS"
    assert result.job_evidence_source == "location"
    assert result.job_evidence == "London, England, United Kingdom"


@pytest.mark.parametrize("description", [
    "This role is based in Bengaluru, India.", "This role is not based in the UK.",
    "This role may be based in the UK.", "This role might be based in India.",
    "This role will not be based in the UK.",
    "Our headquarters are in the UK, but this role is based in India.",
    "Location: UK or India", "You cannot work from the UK.",
    "Remote within India.",
])
def test_location_field_does_not_override_conflicting_conditional_or_negated_description(description):
    result = checks(description, location="London, United Kingdom", locations=["United Kingdom"])["location"]
    assert result.status == "UNKNOWN"


@pytest.mark.parametrize("location", [
    "UK or India", "London, UK / Bengaluru, India", "UK subject to relocation",
    "Not United Kingdom", "Headquarters: London, UK", "Remote",
])
def test_ambiguous_metadata_location_does_not_pass(location):
    assert checks("Build software.", location=location, locations=["United Kingdom"])["location"].status == "UNKNOWN"


def test_metadata_location_and_matching_description_agree():
    assert checks("This role is based in London, UK.", location="London, United Kingdom",
                  locations=["United Kingdom"])["location"].status == "PASS"


@pytest.mark.parametrize("destination", ["London", "London, UK"])
def test_required_commute_can_be_corroborated_by_explicit_structured_city_and_country(destination):
    result = checks(f"This role requires commuting into our {destination} office Mondays through Fridays.",
                    location="London, England, United Kingdom", locations=["United Kingdom"])["location"]
    assert result.status == "PASS"
    assert result.job_evidence_source == "location"
    assert result.job_evidence == "London, England, United Kingdom"


@pytest.mark.parametrize("description", [
    "This role requires commuting into our Berlin office Mondays through Fridays.",
    "This role requires commuting to Germany daily.",
    "You must commute to our office in Germany daily.",
    "This role may require commuting into our London office Mondays through Fridays.",
    "This role does not require commuting into our London office Mondays through Fridays.",
    "This role requires commuting into our office Mondays through Fridays.",
    "This role requires commuting regularly.",
    "Commuting into our London office is not required.",
    "You must commute from Berlin to London daily.",
    "You must commute to our London office and our Berlin office weekly.",
])
def test_metadata_cannot_override_conflicting_conditional_negated_or_unparseable_required_commute(description):
    result = checks(description, location="London, England, United Kingdom", locations=["United Kingdom"])["location"]
    assert result.status == "UNKNOWN"
    assert result.job_evidence_source == "description"
    assert result.job_evidence == description


def test_commute_city_without_corroborating_metadata_does_not_infer_country():
    description = "This role requires commuting into our London office Mondays through Fridays."
    assert checks(description, locations=["United Kingdom"])["location"].status == "UNKNOWN"
    assert checks(description, location="United Kingdom", locations=["United Kingdom"])["location"].status == "UNKNOWN"


def test_ambiguous_commute_city_country_cannot_override_explicit_location():
    description = "This role requires commuting into our London, Ontario office Mondays through Fridays."
    assert checks(description, location="London, United Kingdom", locations=["United Kingdom"])["location"].status == "UNKNOWN"


def test_explicit_required_foreign_commute_without_metadata_fails_country_preference():
    description = "This role requires commuting to Germany daily."
    assert checks(description, locations=["United Kingdom"])["location"].status == "FAIL"


@pytest.mark.parametrize("description", [
    "This role requires working in Germany.", "You must work in Germany.",
    "This role may require working in the UK.", "You may work in the UK.",
    "You cannot work in the UK.", "You must not work in the UK.",
])
def test_explicit_workplace_country_cannot_be_overridden_by_metadata(description):
    assert checks(description, location="London, United Kingdom", locations=["United Kingdom"])["location"].status == "UNKNOWN"


def test_structured_country_mismatch_fails_location_preference():
    assert checks("Build software.", location="Bengaluru, India",
                  locations=["United Kingdom"])["location"].status == "FAIL"


def test_structured_india_location_does_not_create_sponsorship_exemption():
    assert checks("Visa sponsorship is not available.", location="Bengaluru, India",
                  requires_sponsorship=True, sponsorship_exempt_locations=["India"])["sponsorship"].status == "UNKNOWN"


def test_conflicting_metadata_invalidates_description_india_exemption():
    assert checks("Location: Bengaluru, India. Visa sponsorship is not available.",
                  location="London, United Kingdom", requires_sponsorship=True,
                  sponsorship_exempt_locations=["India"])["sponsorship"].status == "UNKNOWN"


def test_structured_location_does_not_promise_sponsorship():
    assert checks("Will you require sponsorship now or in the future?", location="London, United Kingdom",
                  requires_sponsorship=True)["sponsorship"].status == "UNKNOWN"


def test_structured_location_can_reject_sponsorship_offer_for_another_country():
    assert checks("Visa sponsorship is available in Germany.", location="London, United Kingdom",
                  requires_sponsorship=True)["sponsorship"].status == "UNKNOWN"


@pytest.mark.parametrize("description", [
    "Location: London, United Kingdom", "This role is based in Bristol, England.",
    "Location: Remote (UK only)", "This role is fully remote within the UK.",
    "This role is based in London, UK; no relocation is required.",
])
def test_explicit_uk_work_location_matches_anywhere_uk(description):
    assert checks(description, locations=["United Kingdom"])["location"].status == "PASS"


@pytest.mark.parametrize("description", [
    "This role is not based in the United Kingdom.",
    "This role is based in Bengaluru, India.",
    "You cannot work from the UK.",
])
def test_explicit_country_mismatch_or_exclusion_fails(description):
    assert checks(description, locations=["United Kingdom"])["location"].status == "FAIL"


@pytest.mark.parametrize("description", [
    "Our headquarters are in the UK.", "Location: London",
    "Location: United Kingdom or India", "This role may be based in the UK.",
    "Location: Cambridge, United States", "Location: Paris, France or London, UK",
])
def test_unresolved_location_is_not_passed(description):
    expected = "FAIL" if description == "Location: Cambridge, United States" else "UNKNOWN"
    assert checks(description, locations=["United Kingdom"])["location"].status == expected


def test_country_word_inside_city_preference_does_not_allow_other_cities():
    assert checks("Location: Manchester, UK", locations=["London, United Kingdom"])["location"].status == "UNKNOWN"


@pytest.mark.parametrize("description,status", [
    ("This role is fully remote.", "PASS"),
    ("This role is remote with no travel required.", "PASS"),
    ("This role is not remote.", "FAIL"),
    ("Remote working is not available.", "FAIL"),
    ("This role is onsite.", "FAIL"),
    ("This role is hybrid.", "FAIL"),
    ("This role may be remote.", "UNKNOWN"),
    ("This role is remote or hybrid.", "UNKNOWN"),
    ("This role develops remote access software.", "UNKNOWN"),
])
def test_work_patterns_respect_actual_arrangement_and_negation(description, status):
    assert checks(description, work_patterns=["remote"])["work_pattern"].status == status


@pytest.mark.parametrize("statement,status", [
    ("Visa sponsorship is available.", "PASS"),
    ("We offer visa sponsorship.", "PASS"),
    ("We can sponsor skilled worker visas.", "PASS"),
    ("Visa sponsorship is not available.", "FAIL"),
    ("We cannot sponsor visas.", "FAIL"),
    ("We do not offer visa sponsorship.", "FAIL"),
    ("Applicants must not require sponsorship.", "FAIL"),
    ("You must already have the right to work in the UK.", "FAIL"),
    ("Visa sponsorship may be available.", "UNKNOWN"),
    ("Visa sponsorship is available subject to eligibility.", "UNKNOWN"),
    ("Visa sponsorship is available for eligible candidates.", "UNKNOWN"),
    ("Visa sponsorship is available if you already live in the UK.", "UNKNOWN"),
    ("Visa sponsorship is available to applicants with a UK postgraduate degree.", "UNKNOWN"),
    ("Visa sponsorship is available for other roles.", "UNKNOWN"),
    ("Visa sponsorship is available at our US office.", "UNKNOWN"),
    ("Visa sponsorship is available in Germany.", "UNKNOWN"),
    ("You do not need existing right to work.", "UNKNOWN"),
    ("We have a sponsorship licence.", "UNKNOWN"),
])
def test_sponsorship_requires_explicit_suitable_offer(statement, status):
    text = "This role is based in London, UK. " + statement
    assert checks(text, requires_sponsorship=True)["sponsorship"].status == status


def test_conflicting_sponsorship_terms_are_unknown():
    result = checks("Visa sponsorship is available. We cannot sponsor visas.", requires_sponsorship=True)
    assert result["sponsorship"].status == "UNKNOWN"


def test_unrelated_no_clause_cannot_be_read_as_refusal_of_sponsorship():
    result = checks("No prior industry experience is required; visa sponsorship is available.", requires_sponsorship=True)
    assert result["sponsorship"].status != "FAIL"


def test_explicit_india_job_is_exempt_even_without_sponsorship():
    result = checks("This role is based in Bengaluru, India. Visa sponsorship is not available.", requires_sponsorship=True, sponsorship_exempt_locations=["India"])
    assert result["sponsorship"].status == "PASS"
    assert "Bengaluru, India" in result["sponsorship"].job_evidence


@pytest.mark.parametrize("description", [
    "Our headquarters are in India. We cannot sponsor visas.",
    "Location: India or United Kingdom. We cannot sponsor visas.",
    "Location: Bengaluru, India. This is a fully remote role.",
    "Location: Bengaluru, India. This role is remote.",
    "Location: Bengaluru, India. Work pattern: remote.",
    "Location: Remote within India. We cannot sponsor visas.",
    "This role may be based in India. We cannot sponsor visas.",
    "Location: Bengaluru. We cannot sponsor visas.",
])
def test_india_exemption_requires_unambiguous_actual_work_location(description):
    result = checks(description, requires_sponsorship=True, sponsorship_exempt_locations=["India"])
    assert result["sponsorship"].status == "UNKNOWN"


def test_india_headquarters_does_not_exempt_a_uk_vacancy():
    result = checks("Our headquarters are in India. This role is based in London, UK. We cannot sponsor visas.", requires_sponsorship=True, sponsorship_exempt_locations=["India"])
    assert result["sponsorship"].status == "FAIL"


@pytest.mark.parametrize("description,status", [
    ("Annual salary: GBP 60000.", "PASS"),
    ("Salary: £50,000–£70,000 per annum.", "PASS"),
    ("Annual salary: £60–70k.", "PASS"),
    ("Annual salary: £60,000–70k.", "PASS"),
    ("Salary: £40,000 per year.", "FAIL"),
    ("Salary: £60,000.", "UNKNOWN"),
    ("Annual salary: $80,000.", "UNKNOWN"),
    ("Annual salary: USD 80000.", "UNKNOWN"),
    ("Estimated annual salary: £80,000.", "UNKNOWN"),
    ("Salary: £8,000 per month, £96,000 annually.", "UNKNOWN"),
    ("Annual salary: £60000 plus £20000 bonus.", "UNKNOWN"),
    ("Annual salary: £50000–€80000.", "UNKNOWN"),
    ("Annual salary: from £40000.", "UNKNOWN"),
    ("Annual salary: £80000 full-time equivalent for this part-time role.", "UNKNOWN"),
    ("Annual salary: £80000 pro rata.", "UNKNOWN"),
])
def test_salary_needs_explicit_annual_currency_and_unambiguous_base_pay(description, status):
    assert checks(description, minimum_salary=60000)["salary"].status == status


def test_predicted_salary_and_salary_metadata_alone_cannot_pass():
    assert checks("Annual salary: £80,000.", minimum_salary=60000, metadata={"salary_is_predicted": 1})["salary"].status == "UNKNOWN"
    assert checks("Competitive compensation.", minimum_salary=60000, metadata={"salary_min": 80000, "currency": "GBP"})["salary"].status == "UNKNOWN"
    assert checks("Annual salary: £80,000.", minimum_salary=60000, metadata={"salary_is_predicted": "True"})["salary"].status == "UNKNOWN"


def test_custom_constraints_are_unknown_even_if_the_text_seems_to_match():
    result = checks("Visa sponsorship is available.", hard_constraints=["Must offer sponsorship", "Never work in finance"])
    assert set(result) == {"custom_1", "custom_2"}
    assert all(item.status == "UNKNOWN" for item in result.values())


def test_absent_preferences_produce_no_checks():
    assert checks("A job description") == {}
