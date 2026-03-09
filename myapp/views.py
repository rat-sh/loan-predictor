import os
import glob
import numpy as np
import joblib
from django.shortcuts import render

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# Debug log so you can see the path in Docker logs
import sys
print(f"[Lokkhi] MODEL_DIR = {MODEL_DIR}", file=sys.stderr)
print(f"[Lokkhi] PKL count = {len(glob.glob(os.path.join(MODEL_DIR, '*_model.pkl')))}", file=sys.stderr)
TARGET_COL = "Loan_Status"


def get_bank_names():
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*_model.pkl")))
    return [os.path.basename(f).replace("_model.pkl", "").replace("_", " ") for f in files]


def encode_input(user_input, encoders, feature_names):
    row = {}
    for feat in feature_names:
        val = user_input.get(feat)
        if feat in encoders:
            le  = encoders[feat]
            val = str(val) if val is not None else le.classes_[0]
            row[feat] = int(le.transform([val])[0]) if val in le.classes_ else 0
        else:
            row[feat] = float(val or 0)
    return np.array([[row[f] for f in feature_names]])


def predict_one(bank_name, user_input):
    pkl = os.path.join(MODEL_DIR, f"{bank_name.replace(' ', '_')}_model.pkl")
    if not os.path.exists(pkl):
        return {"bank": bank_name, "result": "Model not found", "confidence": None, "accuracy": 0}

    bundle   = joblib.load(pkl)
    X        = encode_input(user_input, bundle["encoders"], bundle["feature_names"]).astype(float)
    pred     = bundle["model"].predict(X)[0]
    conf     = round(float(max(bundle["model"].predict_proba(X)[0])) * 100, 1) if hasattr(bundle["model"], "predict_proba") else None
    loan_le  = bundle["encoders"][TARGET_COL]
    appr_enc = loan_le.transform([[c for c in loan_le.classes_ if str(c).upper() in ("Y", "YES", "APPROVED", "1")][0]])[0]

    return {
        "bank":       bank_name,
        "result":     "Approved" if pred == appr_enc else "Rejected",
        "confidence": conf,
        "accuracy":   round(bundle["accuracy"] * 100, 1)
    }


def predict_all(user_input):
    approved, rejected = [], []
    for pkl in sorted(glob.glob(os.path.join(MODEL_DIR, "*_model.pkl"))):
        bank = os.path.basename(pkl).replace("_model.pkl", "").replace("_", " ")
        r    = predict_one(bank, user_input)
        (approved if r["result"] == "Approved" else rejected).append(r)
    approved.sort(key=lambda x: x["confidence"] or 0, reverse=True)
    rejected.sort(key=lambda x: x["confidence"] or 0, reverse=True)
    return approved, rejected


# ── HOME ──
def home(request):
    return render(request, "home.html")


# ── FORM PAGE ──
def loan_form(request):
    return render(request, "loan_form.html", {
        "bank_names": get_bank_names(),
    })


# ── RESULT PAGE ──
def loan_result(request):
    if request.method != "POST":
        # if someone visits /result/ directly, send them to form
        from django.shortcuts import redirect
        return redirect("/loan/")

    approved, rejected, result_one, mode = [], [], None, None

    user_input = {
        "Gender":            request.POST.get("gender"),
        "Married":           request.POST.get("married"),
        "Dependents":        request.POST.get("dependents"),
        "Education":         request.POST.get("education"),
        "Self_Employed":     request.POST.get("self_employed"),
        "ApplicantIncome":   float(request.POST.get("applicant_income", 0)),
        "CoapplicantIncome": float(request.POST.get("coapplicant_income", 0)),
        "LoanAmount":        float(request.POST.get("loan_amount", 0)),
        "Loan_Amount_Term":  float(request.POST.get("loan_term", 360)),
        "Credit_History":    float(request.POST.get("credit_history", 1)),
        "Property_Area":     request.POST.get("property_area"),
        "Loan_Purpose":      request.POST.get("loan_purpose"),
    }

    if "predict_one" in request.POST:
        mode       = "single"
        result_one = predict_one(request.POST.get("bank_name"), user_input)

    elif "predict_all" in request.POST:
        mode = "all"
        approved, rejected = predict_all(user_input)

    return render(request, "result.html", {
        "result_one": result_one,
        "approved":   approved,
        "rejected":   rejected,
        "mode":       mode,
        "total":      len(approved) + len(rejected),
        "request":    request,
    })