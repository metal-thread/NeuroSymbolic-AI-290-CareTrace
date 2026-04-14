import os
import sys

# Ensure parent and agents directories are in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents")))

from langgraph.types import Command
from triage_state import TriageState, ClinicalState
from logic_safety_agent import logic_safety_agent

def test_er_rule_infant():
    print("Testing ER Rule: Infant < 3 months with fever...")
    cs = ClinicalState(cpg_age=2, cpg_body_temperature=101.0, fever_longer_than_24_hours=False, fever_longer_than_3_days=False)
    state = {"clinical_state": cs}
    result = logic_safety_agent(state)
    assert not isinstance(result, Command)
    assert result['decision']['disposition'] == "Emergency Department Now"
    assert result['decision']['reason'] == "age_and_duration_rule"
    print("✓ Passed")

def test_er_rule_high_temp():
    print("Testing ER Rule: High Temperature Gate...")
    cs = ClinicalState(cpg_age=36, cpg_body_temperature=104.5, fever_longer_than_24_hours=False, fever_longer_than_3_days=False)
    state = {"clinical_state": cs}
    result = logic_safety_agent(state)
    assert not isinstance(result, Command)
    assert result['decision']['disposition'] == "Emergency Department Now"
    assert result['decision']['reason'] == "temperature_hard_gate"
    print("✓ Passed")

def test_home_observation_rule():
    print("Testing Home Observation Rule...")
    cs = ClinicalState(
        cpg_age=72, # 6 years
        cpg_body_temperature=101.8,
        fever_longer_than_24_hours=False,
        fever_longer_than_3_days=False,
        cpg_eating='normal appetite',
        cpg_behavior='playful',
        cpg_comfort_level='good',
        cpg_is_lethargic=False,
        cpg_vomiting=False,
        cpg_seizure=False,
        cpg_rash=False,
        cpg_wetting_diapers=True,
        cpg_dry_mouth=False
    )
    state = {"clinical_state": cs}
    result = logic_safety_agent(state)
    if isinstance(result, Command):
        print(f"  (Received Command instead of result. Missing: {result.update['unknowns']})")
        assert False, "Should have enough data for home observation"
    assert result['decision']['disposition'] == "Home Management"
    assert "acetaminophen" in result['decision']['medications']
    assert "ibuprofen" in result['decision']['medications']
    print("✓ Passed")

def test_missing_info_command():
    print("Testing Missing Information Command (Loopback)...")
    cs = ClinicalState(cpg_age=None, cpg_body_temperature=101.8)
    state = {"clinical_state": cs}
    result = logic_safety_agent(state)
    
    assert isinstance(result, Command)
    assert result.goto == "interpretation_agent"
    assert "cpg_age" in result.update["unknowns"]
    print("✓ Passed")

if __name__ == "__main__":
    try:
        test_er_rule_infant()
        test_er_rule_high_temp()
        test_home_observation_rule()
        test_missing_info_command()
        print("\nAll logic_safety_agent tests passed successfully.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
