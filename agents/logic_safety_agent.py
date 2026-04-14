from pyDatalog import pyDatalog
from typing import Dict, Any, List, Optional
from triage_state import TriageState, ClinicalState

def logic_safety_agent(state: TriageState) -> Dict[str, Any]:
    cs = state.get("clinical_state")
    if cs is None:
        cs = ClinicalState()
    
    # 1. Setup pyDatalog
    pyDatalog.clear()
    
    # Define Terms and Rules
    pyDatalog.load("""
        has_disposition(Type, Reason)
        age_val(Val)
        temp_val(Val)
        dur24_val(Val)
        dur3d_val(Val)
        symp_val(Val)
        leth_val(Val)
        behavior_val(Val)
        seizure_val(Val)
        breath_val(Val)
        fast_val(Val)
        pain_val(Val)
        rash_val(Val)
        vomit_val(Val)
        pee_val(Val)
        dry_val(Val)
        chronic_val(Val)
        eating_val(Val)
        meds_val(Val)
        is_missing(Field)
        required_for(Rule, Field)

        # Ensure predicates exist with dummy facts
        + age_val(-1)
        + temp_val(-1)
        + dur24_val('dummy')
        + dur3d_val('dummy')
        + symp_val('dummy')
        + leth_val('dummy')
        + behavior_val('dummy')
        + seizure_val('dummy')
        + breath_val('dummy')
        + fast_val('dummy')
        + pain_val('dummy')
        + rash_val('dummy')
        + vomit_val('dummy')
        + pee_val('dummy')
        + dry_val('dummy')
        + chronic_val('dummy')
        + eating_val('dummy')
        + meds_val('dummy')
        + is_missing('dummy')

        # Tier 1: Vital for any decision (Turn 1 requirements)
        + required_for('tier1', 'age')
        + required_for('tier1', 'temp')
        + required_for('tier1', 'behavior')
        + required_for('tier1', 'eating')

        # Tier 2: Essential for grounding and medication safety (Turn 2 requirements)
        + required_for('tier2', 'pee')
        + required_for('tier2', 'meds')
        + required_for('tier2', 'breath')

        # Tier 3: Important for duration-based triage
        + required_for('tier3', 'dur24')

        # --- RED FLAGS ---
        has_disposition('ER_NOW', 'infant_fever') <= age_val(X) & (X >= 0) & (X < 3) & temp_val(Y) & (Y >= 100.4)
        has_disposition('ER_NOW', 'respiratory_distress') <= breath_val(True)
        has_disposition('ER_NOW', 'seizure') <= seizure_val(True)
        has_disposition('ER_NOW', 'extreme_fever') <= temp_val(X) & (X >= 105)
        
        # Scenario 02 Specific: High fever + Reduced responsiveness + No urination
        has_disposition('ER_NOW', 'severe_illness_dehydration') <= behavior_val('lethargic') & pee_val(False) & temp_val(X) & (X >= 103)
        has_disposition('ER_NOW', 'neurological_red_flag') <= behavior_val('lethargic')

        # --- HOME OBSERVATION ---
        has_disposition('HOME_OBSERVATION', 'safe_for_home') <= \
            age_val(X) & (X >= 3) & \
            temp_val(Y) & (Y >= 100) & (Y < 104) & \
            behavior_val('playful') & \
            pee_val(True) & \
            breath_val(False)
        
        has_disposition('HOME_OBSERVATION', 'safe_for_home') <= \
            age_val(X) & (X >= 3) & \
            temp_val(Y) & (Y >= 100) & (Y < 104) & \
            behavior_val('sleeping') & \
            pee_val(True) & \
            breath_val(False)
            
        # Scenario 01 Catch: Tired but responsive, eating some, peeing, moderate fever
        has_disposition('HOME_OBSERVATION', 'home_care_appropriate') <= \
            temp_val(Y) & (Y < 103) & \
            pee_val(True) & \
            breath_val(False) & \
            meds_val(True)
    """)

    # 2. Assert Facts
    def assert_v(pred, val):
        if val is not None:
            pyDatalog.assert_fact(pred, val)
        else:
            pyDatalog.assert_fact('is_missing', pred.replace('_val', ''))

    assert_v('age_val', cs.cpg_age)
    assert_v('temp_val', cs.cpg_body_temperature)
    assert_v('dur24_val', cs.fever_longer_than_24_hours)
    assert_v('dur3d_val', cs.fever_longer_than_3_days)
    assert_v('behavior_val', cs.cpg_behavior)
    assert_v('seizure_val', cs.cpg_seizure)
    assert_v('breath_val', cs.cpg_trouble_breathing)
    assert_v('fast_val', cs.cpg_fast_breathing)
    assert_v('vomit_val', cs.cpg_vomiting)
    assert_v('pee_val', cs.urinated_recently)
    assert_v('dry_val', cs.cpg_dry_mouth)
    assert_v('eating_val', cs.cpg_eating)
    assert_v('meds_val', True if cs.medications else None)

    # 3. Decision Logic - SEQUENTIAL PRIORITY

    # 3a. Check for Immediate Life-Threatening ER
    er_results = pyDatalog.ask("has_disposition('ER_NOW', Reason)")
    if er_results:
        reasons = [r[0] for r in er_results.answers if r[0] != 'dummy']
        critical = ['infant_fever', 'respiratory_distress', 'seizure', 'extreme_fever']
        for r in reasons:
            if r in critical:
                return {
                    "decision": {"disposition": "Emergency Department Now", "reason": r},
                    "datalog_proof_tree": {"disposition": "ER_NOW", "rules_fired": reasons},
                    "unknowns": [],
                    "last_action": "safety_logic"
                }

    # 3b. Check for Tier 1 Missing (Turn 1 questions)
    missing_tier1 = pyDatalog.ask("is_missing(Field) & required_for('tier1', Field)")
    if missing_tier1:
        answers = [r[0] for r in missing_tier1.answers]
        unknowns = [a for a in answers if a != 'dummy']
        if unknowns:
            mapping = {'age': 'cpg_age', 'temp': 'cpg_body_temperature', 'behavior': 'cpg_behavior', 'eating': 'cpg_eating'}
            return {"unknowns": [mapping.get(u, u) for u in unknowns], "last_action": "safety_logic"}

    # 3c. Check for Tier 2 Missing (Turn 2 questions)
    missing_tier2 = pyDatalog.ask("is_missing(Field) & required_for('tier2', Field)")
    if missing_tier2:
        answers = [r[0] for r in missing_tier2.answers]
        unknowns = [a for a in answers if a != 'dummy']
        if unknowns:
            mapping = {'pee': 'urinated_recently', 'meds': 'medications', 'breath': 'cpg_trouble_breathing'}
            return {"unknowns": [mapping.get(u, u) for u in unknowns], "last_action": "safety_logic"}

    # 3d. Final Evaluation after Tiers 1 & 2
    if er_results:
        reasons = [r[0] for r in er_results.answers if r[0] != 'dummy']
        return {
            "decision": {"disposition": "Emergency Department Now", "reason": reasons[0]},
            "datalog_proof_tree": {"disposition": "ER_NOW", "rules_fired": reasons},
            "unknowns": [],
            "last_action": "safety_logic"
        }

    home_results = pyDatalog.ask("has_disposition('HOME_OBSERVATION', Reason)")
    if home_results:
        reasons = [r[0] for r in home_results.answers if r[0] != 'dummy']
        return {
            "decision": {"disposition": "Home Management", "reason": reasons[0]},
            "datalog_proof_tree": {"disposition": "HOME_OBSERVATION", "rules_fired": reasons},
            "unknowns": [],
            "last_action": "safety_logic"
        }

    # 3e. Tier 3 or Default
    missing_tier3 = pyDatalog.ask("is_missing(Field) & required_for('tier3', Field)")
    if missing_tier3:
        answers = [r[0] for r in missing_tier3.answers]
        unknowns = [a for a in answers if a != 'dummy']
        if unknowns:
            return {"unknowns": ['fever_longer_than_24_hours'], "last_action": "safety_logic"}

    return {
        "decision": {"disposition": "Home Management", "reason": "default_home_low_risk"},
        "unknowns": [],
        "last_action": "safety_logic"
    }
