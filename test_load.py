import joblib

print("Loading vectorizer...")
v = joblib.load("tfidf.joblib")
print("✓ Vectorizer loaded")

print("Loading module model...")
m = joblib.load("sap_module_model.joblib")
print("✓ Module model loaded")

print("Loading priority model...")
p = joblib.load("sap_priority_model.joblib")
print("✓ Priority model loaded")

print("Everything loaded successfully!")