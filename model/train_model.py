# ================================
# IMPORTS
# ================================
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import os
import shutil
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline


# ================================
# CONFIG ASSETS
# ================================
ASSETS_DIR = "assets"

def reset_assets_folder():
    if os.path.exists(ASSETS_DIR):
        shutil.rmtree(ASSETS_DIR)
    os.makedirs(ASSETS_DIR)

def save_plot(filename):
    path = os.path.join(ASSETS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

# limpa pasta antes de rodar
reset_assets_folder()


# ================================
# LOAD
# ================================
df = pd.read_csv("data/cardio_train.csv", sep=";")

if "id" in df.columns:
    df = df.drop(columns=["id"])


# ================================
# FEATURE ENGINEERING
# ================================
df["age_years"] = df["age"] / 365

# ================================
# LIMPEZA
# ================================
df = df[(df["age_years"] >= 18) & (df["age_years"] <= 100)]
df = df[(df["height"] >= 140) & (df["height"] <= 210)]
df = df[(df["weight"] >= 40) & (df["weight"] <= 200)]
df = df[(df["ap_hi"] >= 90) & (df["ap_hi"] <= 200)]
df = df[(df["ap_lo"] >= 60) & (df["ap_lo"] <= 120)]

# ================================
# NOVAS FEATURES
# ================================
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df = df[(df["bmi"] >= 15) & (df["bmi"] <= 50)]
df["pressure_age"] = df["ap_hi"] * df["age_years"]

# ================================
# FEATURES FINAIS
# ================================
features_final = [
    "ap_hi", "ap_lo", "age_years", "bmi",
    "cholesterol", "gluc", "smoke",
    "alco", "active", "pressure_age"
]

X = df[features_final]
y = df["cardio"]


# ================================
# SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================================
# PIPELINE
# ================================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, X.columns)
])

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    ))
])


# ================================
# GRID SEARCH
# ================================
param_grid = {
    "model__n_estimators": [300, 500],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.03, 0.05],
    "model__subsample": [0.7, 0.8],
    "model__colsample_bytree": [0.7, 0.8]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
pipeline = grid_search.best_estimator_


# ================================
# THRESHOLD OTIMIZADO
# ================================
y_prob = pipeline.predict_proba(X_test)[:, 1]

best_threshold = 0
best_precision = 0

for t in np.arange(0.1, 0.9, 0.01):
    y_pred_temp = (y_prob > t).astype(int)

    recall = recall_score(y_test, y_pred_temp)
    precision = precision_score(y_test, y_pred_temp)

    if recall >= 0.80 and precision > best_precision:
        best_precision = precision
        best_threshold = t

print(f"Melhor threshold: {best_threshold:.2f}")


# ================================
# RESULTADOS
# ================================
y_pred = (y_prob > best_threshold).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print("F1:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))


# ================================
# ROC
# ================================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.title("Curva ROC")
plt.legend()
save_plot("roc_curve.png")


# ================================
# PRECISION-RECALL
# ================================
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(recall_vals, precision_vals)
plt.title("Precision-Recall")
save_plot("precision_recall.png")


# ================================
# MATRIZ DE CONFUSÃO
# ================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
save_plot("confusion_matrix.png")


# ================================
# CALIBRATION
# ================================
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.figure(figsize=(8,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], linestyle='--')
plt.title("Calibration Curve")
save_plot("calibration_curve.png")


# ================================
# FEATURE IMPORTANCE
# ================================
xgb_model = pipeline.named_steps["model"]

importances = xgb_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Importância das Features")
save_plot("feature_importance.png")


# ================================
# SHAP
# ================================
X_test_transformed = pipeline.named_steps["prep"].transform(X_test)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=X.columns,
    show=False
)

plt.savefig(os.path.join(ASSETS_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
plt.close()


# ================================
# SAVE MODEL
# ================================
with open("model/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Modelo salvo com sucesso!")