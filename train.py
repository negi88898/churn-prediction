"""
Train and compare churn-prediction models.

Usage:  python train.py
Steps:  load data -> clean -> split -> preprocess -> train 3 models -> evaluate
        -> save best model (models/churn_model.joblib) + charts (outputs/)
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

for _d in ("models", "outputs"):
    os.makedirs(_d, exist_ok=True)
DATA_PATH = "data/telco_churn.csv"
RANDOM_STATE = 42

# ---------- 1. Load & clean ----------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

df = df.drop(columns=["customerID"])
# In the real Telco data TotalCharges has blank strings -> convert to numbers
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = (df["Churn"] == "Yes").astype(int)

print(f"Churn rate: {df['Churn'].mean():.1%}")

# ---------- 2. EDA charts ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df["Churn"].value_counts().rename({0: "Stayed", 1: "Churned"}).plot.bar(ax=axes[0], color=["#4C9F70", "#D9534F"])
axes[0].set_title("Churn distribution")
df.groupby("Contract")["Churn"].mean().plot.bar(ax=axes[1], color="#5B8DEF")
axes[1].set_title("Churn rate by contract"); axes[1].set_ylabel("Churn rate")
df.boxplot(column="tenure", by="Churn", ax=axes[2]); axes[2].set_title("Tenure vs churn"); plt.suptitle("")
plt.tight_layout(); plt.savefig("outputs/eda.png", dpi=120); plt.close()

# ---------- 3. Split ----------
X, y = df.drop(columns=["Churn"]), df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ---------- 4. Models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

results, fitted = {}, {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted[name] = pipe
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))

res_df = pd.DataFrame(results).T.round(4)
print("\nModel comparison:\n", res_df)
res_df.to_csv("outputs/model_comparison.csv")

# ---------- 5. Pick best (by ROC-AUC) ----------
best_name = res_df["roc_auc"].idxmax()
best = fitted[best_name]
print(f"\nBest model: {best_name}")
joblib.dump(best, "models/churn_model.joblib")
json.dump({"best_model": best_name, "metrics": results[best_name]},
          open("models/model_info.json", "w"), indent=2)

# ---------- 6. Charts for best model ----------
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Stayed", "Churned"]).plot(ax=ax, cmap="Blues")
ax.set_title(f"Confusion matrix - {best_name}")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4); ax.set_title("ROC curves")
plt.tight_layout(); plt.savefig("outputs/roc_curves.png", dpi=120); plt.close()

# Feature importance (works for Logistic Regression and tree models)
feat_names = best.named_steps["prep"].get_feature_names_out()
mdl = best.named_steps["model"]
imp = np.abs(mdl.coef_[0]) if hasattr(mdl, "coef_") else mdl.feature_importances_
top = pd.Series(imp, index=feat_names).sort_values().tail(12)
top.index = [i.replace("cat__", "").replace("num__", "") for i in top.index]
fig, ax = plt.subplots(figsize=(7, 5))
top.plot.barh(ax=ax, color="#5B8DEF"); ax.set_title(f"Top 12 churn drivers - {best_name}")
plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=120); plt.close()

print("\nSaved: models/churn_model.joblib, outputs/*.png, outputs/model_comparison.csv")
