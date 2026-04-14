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
        cranky_val(Val)
        sleepy_val(Val)
        seizure_val(Val)
        breath_val(Val)
        fast_val(Val)
        pain_val(Val)
        rash_val(Val)
        vomit_val(Val)
        vomit24_val(Val)
        pee_val(Val)
        dry_val(Val)
        chronic_val(Val)
        eating_val(Val)
        sleep_val(Val)
        ok_val(Val)
        behavior_val(Val)
        is_missing(Field)
        required_for(Rule, Field)

        # Ensure predicates exist with dummy facts
        + age_val(-1)
        + temp_val(-1)
        + dur24_val('dummy')
        + dur3d_val('dummy')
        + symp_val('dummy')
        + leth_val('dummy')
        + cranky_val('dummy')
        + sleepy_val('dummy')
        + seizure_val('dummy')
        + breath_val('dummy')
        + fast_val('dummy')
        + pain_val('dummy')
        + rash_val('dummy')
        + vomit_val('dummy')
        + vomit24_val('dummy')
        + pee_val('dummy')
        + dry_val('dummy')
        + chronic_val('dummy')
        + eating_val('dummy')
        + sleep_val('dummy')
        + ok_val('dummy')
        + behavior_val('dummy')
        + is_missing('dummy')

        # Mandatory Fields for a Complete Clinical Picture
        + required_for('complete_assessment', 'age')
        + required_for('complete_assessment', 'temp')
        + required_for('complete_assessment', 'dur24')
        + required_for('complete_assessment', 'dur3d')
        + required_for('complete_assessment', 'behavior')
        + required_for('complete_assessment', 'eating')
        + required_for('complete_assessment', 'pee')
        + required_for('complete_assessment', 'dry')

        # ER NOW Rules
        has_disposition('ER_NOW', 'infant_fever') <= age_val(X) & (X >= 0) & (X < 3) & temp_val(Y) & (Y >= 100.4)
        has_disposition('ER_NOW', 'prolonged_fever_toddler') <= age_val(X) & (X >= 3) & (X <= 24) & dur24_val(True) & symp_val(False)
        has_disposition('ER_NOW', 'prolonged_fever_generic') <= dur3d_val(True)
        has_disposition('ER_NOW', 'high_fever_gate') <= temp_val(X) & (X >= 104)
        has_disposition('ER_NOW', 'neurological_red_flag') <= leth_val(True)
        has_disposition('ER_NOW', 'neurological_red_flag') <= cranky_val(True)
        has_disposition('ER_NOW', 'neurological_red_flag') <= sleepy_val(True)
        has_disposition('ER_NOW', 'neurological_red_flag') <= seizure_val(True)
        has_disposition('ER_NOW', 'respiratory_distress') <= breath_val(True)
        has_disposition('ER_NOW', 'respiratory_distress') <= fast_val(True)
        has_disposition('ER_NOW', 'severe_pain') <= pain_val(True)
        has_disposition('ER_NOW', 'unexplained_rash') <= rash_val(True)
        has_disposition('ER_NOW', 'severe_vomiting') <= vomit_val(True)
        has_disposition('ER_NOW', 'severe_vomiting') <= vomit24_val(True)
        has_disposition('ER_NOW', 'dehydration') <= pee_val(False)
        has_disposition('ER_NOW', 'dehydration') <= dry_val(True)
        has_disposition('ER_NOW', 'underlying_condition') <= chronic_val(True)

        # HOME OBSERVATION Rule
        has_disposition('HOME_OBSERVATION', 'safe_for_home') <= eating_val('normal appetite') & sleep_val(True) & ok_val(True) & behavior_val('playful') & age_val(X) & (X > 3) & temp_val(Y) & (Y >= 0) & (Y < 104) & dur3d_val(False)
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
    assert_v('symp_val', (len(cs.cpg_accompanying_symptoms) > 0 or cs.cpg_vomiting or cs.cpg_seizure or cs.cpg_rash) if any(v is not None for v in [cs.cpg_vomiting, cs.cpg_seizure, cs.cpg_rash]) else None)
    assert_v('leth_val', cs.cpg_is_lethargic)
    assert_v('behavior_val', cs.cpg_behavior)
    assert_v('cranky_val', cs.cpg_behavior == 'cranky' if cs.cpg_behavior else None)
    assert_v('sleepy_val', cs.cpg_behavior == 'sleeping' if cs.cpg_behavior else None)
    assert_v('seizure_val', cs.cpg_seizure)
    assert_v('breath_val', cs.cpg_trouble_breathing)
    assert_v('fast_val', cs.cpg_fast_breathing)
    assert_v('pain_val', cs.cpg_pain)
    assert_v('rash_val', cs.cpg_rash)
    assert_v('vomit_val', cs.cpg_vomiting)
    assert_v('vomit24_val', (cs.cpg_vomiting and cs.fever_longer_than_24_hours) if cs.cpg_vomiting is not None and cs.fever_longer_than_24_hours is not None else None)
    assert_v('pee_val', cs.cpg_wetting_diapers)
    assert_v('dry_val', cs.cpg_dry_mouth)
    assert_v('chronic_val', cs.cpg_has_chronic_condition)
    assert_v('eating_val', cs.cpg_eating)
    assert_v('sleep_val', True) 
    assert_v('ok_val', cs.cpg_comfort_level == 'good' or cs.cpg_comfort_level == 'ok' if cs.cpg_comfort_level else None)

    # 3. Decision Logic
    # 3a. Check ER triggers FIRST for safety
    er_results = pyDatalog.ask("has_disposition('ER_NOW', Reason)")
    if er_results:
        reasons = [r[0] for r in er_results.answers if r[0] != 'dummy']
        if reasons:
            return {
                "decision": {"disposition": "Emergency Department Now", "reason": reasons[0]},
                "datalog_proof_tree": {"disposition": "ER_NOW", "rules_fired": reasons},
                "unknowns": [],
                "last_action": "safety_logic"
            }

    # 3b. If no immediate ER triggers, check for missing data to ensure a complete picture
    missing_results = pyDatalog.ask("is_missing(Field) & required_for('complete_assessment', Field)")
    if missing_results:
        answers = [r[0] for r in missing_results.answers]
        unknowns = [a for a in answers if a != 'dummy']
        if unknowns:
            mapping = {
                'age': 'cpg_age', 'temp': 'cpg_body_temperature',
                'dur24': 'fever_longer_than_24_hours', 'dur3d': 'fever_longer_than_3_days',
                'behavior': 'cpg_behavior', 'eating': 'cpg_eating',
                'pee': 'cpg_wetting_diapers', 'dry': 'cpg_dry_mouth'
            }
            mapped_unknowns = [mapping.get(u, u) for u in unknowns]
            return {"unknowns": mapped_unknowns, "last_action": "safety_logic"}

    # 3c. If data is complete, check Home triggers
    home_results = pyDatalog.ask("has_disposition('HOME_OBSERVATION', Reason)")
    if home_results:
        reasons = [r[0] for r in home_results.answers if r[0] != 'dummy']
        if reasons:
            medications = []
            if cs.cpg_age and cs.cpg_age > 3: medications.append("acetaminophen")
            if cs.cpg_age and cs.cpg_age > 6: medications.append("ibuprofen")
            return {
                "decision": {"disposition": "Home Management", "reason": reasons[0], "medications": medications},
                "datalog_proof_tree": {"disposition": "HOME_OBSERVATION", "rules_fired": reasons, "medications": medications},
                "unknowns": [],
                "last_action": "safety_logic"
            }

    # 3c. Default to ER if inconclusive
    return {
        "decision": {"disposition": "Emergency Department Now", "reason": "inconclusive_assessment"},
        "datalog_proof_tree": {"disposition": "DEFAULT_ER", "reason": "No rules matched"},
        "unknowns": [],
        "last_action": "safety_logic"
    }
