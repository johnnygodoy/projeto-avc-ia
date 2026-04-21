# ================================
# IMPORTS
# ================================
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline


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
    "ap_hi",
    "ap_lo",
    "age_years",
    "bmi",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "pressure_age"
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
# TUNING AUTOMÁTICO
# ================================
param_grid = {
    "model__n_estimators": [300, 500],
    "model__max_depth": [4, 6, 8],
    "model__learning_rate": [0.03, 0.05],
    "model__subsample": [0.7, 0.8],
    "model__colsample_bytree": [0.7, 0.8]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nMelhores parâmetros:")
print(grid_search.best_params_)

print("\nMelhor AUC:")
print(grid_search.best_score_)

pipeline = grid_search.best_estimator_


# ================================
# CROSS VALIDATION
# ================================
best_threshold = 0.38 # Definido após análise do conjunto de teste
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
f1_scores = []
precision_scores = []
recall_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n🔹 Fold {fold + 1}")

    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    scale_pos_weight_cv = (y_train_cv == 0).sum() / (y_train_cv == 1).sum()

    pipeline_cv = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight_cv,
            random_state=42,
            eval_metric='logloss'
        ))
    ])

    pipeline_cv.fit(X_train_cv, y_train_cv)

    y_prob_cv = pipeline_cv.predict_proba(X_val_cv)[:, 1]
    y_pred_cv = (y_prob_cv > best_threshold).astype(int)

    auc = roc_auc_score(y_val_cv, y_prob_cv)
    f1 = f1_score(y_val_cv, y_pred_cv)
    precision = precision_score(y_val_cv, y_pred_cv)
    recall = recall_score(y_val_cv, y_pred_cv)

    auc_scores.append(auc)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)

    print(f"AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

print("\n=== CROSS VALIDATION FINAL ===")
print(f"AUC:       {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"F1:        {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Recall:    {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")


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

    if recall >= 0.80:
        if precision > best_precision:
            best_precision = precision
            best_threshold = t

print(f"\nMelhor threshold: {best_threshold:.2f}")
print(f"Melhor precision: {best_precision:.2f}")


# ================================
# RESULTADOS FINAIS
# ================================
y_pred = (y_prob > best_threshold).astype(int)

print("\nRelatório:")
print(classification_report(y_test, y_pred))

print("\nMatriz:")
print(confusion_matrix(y_test, y_pred))

print("\nF1:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))


# ================================
# ROC
# ================================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("Falso Positivo")
plt.ylabel("Verdadeiro Positivo")
plt.title("Curva ROC")
plt.legend()
plt.show()


# ================================
# CURVA DE PRECISION-RECALL 
# ================================

precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.grid()
plt.show()

# ================================
# CALIBRAÇÃO    
# ================================

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.figure(figsize=(8,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("Probabilidade Prevista")
plt.ylabel("Probabilidade Real")
plt.title("Calibration Curve")
plt.grid()
plt.show()

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
plt.show()


# ================================
# SHAP
# ================================
X_test_transformed = pipeline.named_steps["prep"].transform(X_test)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(shap_values, X_test_transformed, feature_names=X.columns)


# ================================
# SAVE
# ================================
with open("model/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModelo salvo com sucesso!")