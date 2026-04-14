import os
import sys

# Ensure parent and agents directories are in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents")))

from triage_state import TriageState, ClinicalState
from logic_safety_agent import logic_safety_agent

def test_er_rule_infant():
    print("Testing ER Rule: Infant < 3 months with fever...")
    # Add mandatory fields to avoid 'unknowns' loopback
    cs = ClinicalState(
        cpg_age=2, cpg_body_temperature=101.0, 
        fever_longer_than_24_hours=False, fever_longer_than_3_days=False,
        cpg_behavior='playful', cpg_eating='normal appetite',
        cpg_wetting_diapers=True, cpg_dry_mouth=False
    )
    state = {"clinical_state": cs}
    result = logic_safety_agent(state)
    assert result['decision']['disposition'] == "Emergency Department Now"
    assert result['decision']['reason'] == "infant_fever"
    print("✓ Passed")

def test_er_rule_high_temp():
    print("Testing ER Rule: High Temperature Gate...")
    cs = ClinicalState(
        cpg_age=36, cpg_body_temperature=104.5, 
        fever_longer_than_24_hours=False, fever_longer_than_3_days=False,
        cpg_behavior='playful', cpg_eating='normal appetite',
        cpg_wetting_diapers=True, cpg_dry_mouth=False
    )
    state = {"clinical_state": cs}
    result = logic_safety_agent(state)
    assert result['decision']['disposition'] == "Emergency Department Now"
    assert result['decision']['reason'] == "high_fever_gate"
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
    assert result['decision']['disposition'] == "Home Management"
    assert "acetaminophen" in result['decision']['medications']
    assert "ibuprofen" in result['decision']['medications']
    print("✓ Passed")

def test_missing_info():
    print("Testing Missing Information (unknowns)...")
    cs = ClinicalState(cpg_age=None, cpg_body_temperature=101.8)
    state = {"clinical_state": cs}
    result = logic_safety_agent(state)
    assert "unknowns" in result
    assert "cpg_age" in result['unknowns']
    print("✓ Passed")

if __name__ == "__main__":
    try:
        test_er_rule_infant()
        test_er_rule_high_temp()
        test_home_observation_rule()
        test_missing_info()
        print("\nAll logic_safety_agent tests passed successfully.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
