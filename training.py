import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
#import mlflow
import joblib

# load dataset
df = pd.read_csv(r"C:\Users\mehant A R\Downloads\sap_ticket_dataset_20000.csv")
df.columns = df.columns.str.strip().str.lower()

X = df["description"].astype(str)
y_module = df["sap_module"].astype(str)
y_priority = df["priority"].astype(str)

# split once so both tasks use same rows
X_train, X_test, ymod_train, ymod_test, ypri_train, ypri_test = train_test_split(
    X, y_module, y_priority,
    test_size=0.2,
    stratify=y_module,
    random_state=42
)
#mlflow.set_experiment("sap_ticket_multi_task")

# TF-IDF
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2,10),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

best_module_model = None
best_priority_model = None
best_module_f1 = 0
best_priority_f1 = 0

for C in [0.1, 1, 5, 10, 20]:
    #with mlflow.start_run(run_name=f"C={C}"):

        # ----- MODULE MODEL -----
        mod_clf = LinearSVC(C=C)
        mod_clf.fit(X_train_vec, ymod_train)
        mod_pred = mod_clf.predict(X_test_vec)

        mod_f1 = f1_score(ymod_test, mod_pred, average="weighted")
        mod_acc = accuracy_score(ymod_test, mod_pred)

        #mlflow.log_metric("module_f1", mod_f1)
        #mlflow.log_metric("module_accuracy", mod_acc)

        if mod_f1 > best_module_f1:
            best_module_f1 = mod_f1
            best_module_model = mod_clf

        # ----- PRIORITY MODEL -----
        pri_clf = LinearSVC(C=C)
        pri_clf.fit(X_train_vec, ypri_train)
        pri_pred = pri_clf.predict(X_test_vec)

        pri_f1 = f1_score(ypri_test, pri_pred, average="weighted")
        pri_acc = accuracy_score(ypri_test, pri_pred)

        #mlflow.log_metric("priority_f1", pri_f1)
        #mlflow.log_metric("priority_accuracy", pri_acc)

        if pri_f1 > best_priority_f1:
            best_priority_f1 = pri_f1
            best_priority_model = pri_clf

# save all
joblib.dump(best_module_model, "sap_module_model.joblib")
joblib.dump(best_priority_model, "sap_priority_model.joblib")
joblib.dump(vectorizer, "tfidf.joblib")

print("Best Module F1:", best_module_f1)
print("Best Priority F1:", best_priority_f1)