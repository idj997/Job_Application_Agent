"""Offline prompt routing/contract checks, not evidence of model compliance."""
from app.applications.cv_critique import CRITIQUE, CRITIQUE_SOURCE_RULES, SCOPED_CRITIQUE


def test_full_and_scoped_prompts_share_the_same_source_contract():
    assert CRITIQUE.startswith(CRITIQUE_SOURCE_RULES)
    assert SCOPED_CRITIQUE.startswith(CRITIQUE_SOURCE_RULES)
    assert "complete master_cv_sources" in CRITIQUE
    assert "entry-only experiment, NOT a complete CV audit" in SCOPED_CRITIQUE
    assert "exactly the supplied section and entry_index" in SCOPED_CRITIQUE


def test_source_comparison_precedes_scoring_and_copywriting_is_not_the_critic_task():
    assert "displayed draft is UNVERIFIED" in CRITIQUE_SOURCE_RULES
    assert "CHECK BEFORE SCORING" in CRITIQUE_SOURCE_RULES
    assert "short EDIT INSTRUCTION" in CRITIQUE_SOURCE_RULES
    assert "Do not compose a\nreplacement CV bullet" in CRITIQUE_SOURCE_RULES
    assert "Draft '<exact disputed words>'; source <real ID> '<exact source words>'" in CRITIQUE_SOURCE_RULES


def test_numeric_scope_and_output_consistency_rules_are_explicit():
    for rule in ("calculate new percentages", "average/median", "contributed to",
                 "extended an existing", "without compromising performance",
                 "Any suggestion means revision_needed=true", "suggestions=[]",
                 "Leave improvements=[]", "JSON null, never CV text"):
        assert rule in CRITIQUE_SOURCE_RULES


def test_examples_are_not_candidate_evidence_and_teach_metric_not_number_matching():
    assert "NEVER candidate facts/evidence" in CRITIQUE_SOURCE_RULES
    assert "number matches\n  but the measured quantity does not" in CRITIQUE_SOURCE_RULES
    assert "source-only achievement omitted from the draft is NOT an unsupported" in CRITIQUE_SOURCE_RULES
    assert "test name is NOT a factual defect" in CRITIQUE_SOURCE_RULES
