from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models
module_model = joblib.load("sap_module_model.joblib")
priority_model = joblib.load("sap_priority_model.joblib")
vectorizer = joblib.load("tfidf.joblib")


def rule_priority(text):
    t = text.lower()

    if any(w in t for w in [
        "system down","production down","business stopped","all users blocked"
    ]):
        return "Critical"

    if any(w in t for w in [
        "blocking","production","month end","finance close","urgent"
    ]):
        return "High"

    if any(w in t for w in [
        "failed","error","cannot","unable","issue"
    ]):
        return "Medium"

    if any(w in t for w in [
        "cosmetic","enhancement","request","clarification"
    ]):
        return "Low"

    return "Not Triggered"


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        text = request.form["ticket"]

        vec = vectorizer.transform([text.lower().strip()])

        module = module_model.predict(vec)[0]

        priority = priority_model.predict(vec)[0]

        scores = priority_model.decision_function(vec)[0]
        classes = priority_model.classes_

        idx = list(classes).index(priority)

        confidence = round(
            100 / (1 + np.exp(-scores[idx])), 2
        )

        rule = rule_priority(text)

        if rule == priority:
            status = "Match"
        elif rule == "Not Triggered":
            status = "No Rule Triggered"
        else:
            status = "Mismatch"

        result = {
            "module": module,
            "priority": priority,
            "confidence": confidence,
            "rule": rule,
            "status": status
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)