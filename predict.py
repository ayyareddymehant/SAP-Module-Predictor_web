# ==========================================
# SAP PRIORITY PREDICTOR
# RELIABLE ML + RULE COMPARISON
# ==========================================

import joblib
import numpy as np

# LOAD
module_model = joblib.load("sap_module_model.joblib")
priority_model = joblib.load("sap_priority_model.joblib")
vectorizer = joblib.load("tfidf.joblib")

# ----- ORIGINAL HARD CODE -----
def rule_priority(text: str):
    t = text.lower()

    if any(w in t for w in [
        "system down","production down","business stopped","all users blocked"
    ]):
        return "Critical"

    if any(w in t for w in [
        "blocking","production","month end","finance close","urgent"
    ]):
        return "High"

    if any(w in t for w in ["failed","error","cannot","unable","issue"]):
        return "Medium"

    if any(w in t for w in ["cosmetic","enhancement","request","clarification"]):
        return "Low"

    return None


print("\nSAP Ticket Predictor (type 'quit' to exit)\n")

while True:
    text = input("Enter ticket description: ")
    if text.lower() == "quit":
        break

    clean = text.lower().strip()
    vec = vectorizer.transform([clean])

    # ===== MODULE (ML) =====
    module_pred = module_model.predict(vec)[0]

    # ===== PRIORITY (ML NATIVE) =====
    ml_priority = priority_model.predict(vec)[0]

    # confidence from margin
    scores = priority_model.decision_function(vec)[0]
    classes = priority_model.classes_

    class_index = list(classes).index(ml_priority)
    margin = scores[class_index]

    # normalize margin → 0-100 confidence
    conf = round(100 / (1 + np.exp(-margin)), 2)

    # ===== RULE =====
    rule_pri = rule_priority(text)

    # ===== COMPARISON =====
    if rule_pri is None:
        status = "No rule triggered"
    elif rule_pri == ml_priority:
        status = "Match"
    else:
        status = "Mismatch"

    # ===== OUTPUT =====
    print("\nPredicted Module :", module_pred)
    print(f"ML Priority      : {ml_priority} ({conf}%)")
    print(f"Rule Priority    : {rule_pri}")

    if status == "Mismatch":
        print("⚠ ML vs Rule mismatch")

    print(f"Final Priority   : {ml_priority} ({conf}%)")
    print("-" * 50)