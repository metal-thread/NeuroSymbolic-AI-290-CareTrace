from pyDatalog import pyDatalog
from typing import Dict, Any, List, Optional
from langgraph.types import Command
from triage_state import TriageState, ClinicalState

def logic_safety_agent(state: TriageState) -> Dict[str, Any] | Command:
    cs = state.get("clinical_state")
    
    # 1. Setup pyDatalog
    pyDatalog.clear()
    
    # Define Terms and Rules via string
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

        # Ensure all predicates exist
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

        # Facts
        + required_for('complete_assessment', 'age')
        + required_for('complete_assessment', 'temp')
        + required_for('complete_assessment', 'dur24')
        + required_for('complete_assessment', 'dur3d')
        + required_for('complete_assessment', 'behavior')
        + required_for('complete_assessment', 'eating')
        + required_for('complete_assessment', 'pee')
        + required_for('complete_assessment', 'dry')

        # Rules
        has_disposition('ER_NOW', 'age_and_duration_rule') <= age_val(X) & (X >= 0) & (X < 3) & temp_val(Y) & (Y >= 100.4)
        has_disposition('ER_NOW', 'age_and_duration_rule') <= age_val(X) & (X >= 3) & (X <= 24) & dur24_val(True) & symp_val(False)
        has_disposition('ER_NOW', 'age_and_duration_rule') <= dur3d_val(True)
        has_disposition('ER_NOW', 'temperature_hard_gate') <= temp_val(X) & (X >= 104)
        has_disposition('ER_NOW', 'behavioral_and_neurological_red_flag') <= leth_val(True)
        has_disposition('ER_NOW', 'behavioral_and_neurological_red_flag') <= cranky_val(True)
        has_disposition('ER_NOW', 'behavioral_and_neurological_red_flag') <= sleepy_val(True)
        has_disposition('ER_NOW', 'behavioral_and_neurological_red_flag') <= seizure_val(True)
        has_disposition('ER_NOW', 'physical_symptoms_and_pain') <= breath_val(True)
        has_disposition('ER_NOW', 'physical_symptoms_and_pain') <= fast_val(True)
        has_disposition('ER_NOW', 'physical_symptoms_and_pain') <= pain_val(True)
        has_disposition('ER_NOW', 'physical_symptoms_and_pain') <= rash_val(True)
        has_disposition('ER_NOW', 'physical_symptoms_and_pain') <= vomit_val(True)
        has_disposition('ER_NOW', 'physical_symptoms_and_pain') <= vomit24_val(True)
        has_disposition('ER_NOW', 'dehydration_hard_gates') <= pee_val(False)
        has_disposition('ER_NOW', 'dehydration_hard_gates') <= dry_val(True)
        has_disposition('ER_NOW', 'underlying_conditions') <= chronic_val(True)

        has_disposition('HOME_OBSERVATION', 'home_observation') <= eating_val('normal appetite') & sleep_val(True) & ok_val(True) & behavior_val('playful') & age_val(X) & (X > 3) & temp_val(Y) & (Y >= 0) & (Y < 104) & dur3d_val(False)
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

    # 3. Query Results
    # ER NOW check first (highest priority)
    er_results = pyDatalog.ask("has_disposition('ER_NOW', Reason)")
    if er_results:
        reasons = [r[0] for r in er_results.answers if r[0] != 'dummy']
        if reasons:
            reason = reasons[0]
            return {
                "decision": {"disposition": "Emergency Department Now", "reason": reason},
                "datalog_proof_tree": {"disposition": "ER_NOW", "rules_fired": reasons}
            }

    # 4. Check for Missing Data if no ER triggers found
    missing_query = "is_missing(Field) & required_for('complete_assessment', Field)"
    missing_results = pyDatalog.ask(missing_query)
    if missing_results:
        answers = [r[0] for r in missing_results.answers]
        unknowns = [a for a in answers if a != 'dummy']
        if unknowns:
            mapping = {
                'age': 'cpg_age',
                'temp': 'cpg_body_temperature',
                'dur24': 'fever_longer_than_24_hours',
                'dur3d': 'fever_longer_than_3_days',
                'behavior': 'cpg_behavior',
                'eating': 'cpg_eating',
                'pee': 'cpg_wetting_diapers',
                'dry': 'cpg_dry_mouth'
            }
            mapped_unknowns = [mapping.get(u, u) for u in unknowns]
            return Command(
                update={"unknowns": mapped_unknowns},
                goto="interpretation_agent"
            )

    # 5. Home Observation check
    home_results = pyDatalog.ask("has_disposition('HOME_OBSERVATION', Reason)")
    if home_results:
        reasons = [r[0] for r in home_results.answers if r[0] != 'dummy']
        if reasons:
            reason = reasons[0]
            medications = []
            if cs.cpg_age > 3: medications.append("acetaminophen")
            if cs.cpg_age > 6: medications.append("ibuprofen")
            
            return {
                "decision": {"disposition": "Home Management", "reason": reason, "medications": medications},
                "datalog_proof_tree": {"disposition": "HOME_OBSERVATION", "rules_fired": reasons, "medications": medications}
            }

    # Safety Default
    return {
        "decision": {"disposition": "Emergency Department Now", "reason": "Inconclusive assessment - escalating for safety"},
        "datalog_proof_tree": {"disposition": "DEFAULT_ER", "reason": "No rules matched"}
    }
