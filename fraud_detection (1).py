# ============================================================
#  Project 2 – Fraud Detection Pipeline
#  DecodeLabs Industrial Training | Batch 2026
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ─────────────────────────────────────────────
# 1. SIMULATE IMBALANCED FINANCIAL DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("  FRAUD DETECTION PIPELINE")
print("=" * 60)

np.random.seed(42)

X, y = make_classification(
    n_samples=20_000,
    n_features=29,          # V1-V28 + Amount
    n_informative=15,
    n_redundant=5,
    weights=[0.9983, 0.0017],   # 99.83% legit / 0.17% fraud
    flip_y=0,
    random_state=42
)

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
df = pd.DataFrame(X, columns=feature_cols)
df["Class"] = y

print(f"\n[DATA] Total transactions  : {len(df):,}")
print(f"[DATA] Legitimate (0)      : {(df.Class==0).sum():,}  ({(df.Class==0).mean()*100:.2f}%)")
print(f"[DATA] Fraudulent (1)      : {(df.Class==1).sum():,}  ({(df.Class==1).mean()*100:.2f}%)")


# ─────────────────────────────────────────────
# 2. STRATIFIED TRAIN / TEST SPLIT  (BEFORE SMOTE!)
# ─────────────────────────────────────────────
X_data = df[feature_cols]
y_data = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data,
    test_size=0.2,
    random_state=42,
    stratify=y_data        # preserves class ratio in both splits
)

print(f"\n[SPLIT] Train fraud cases  : {y_train.sum()}")
print(f"[SPLIT] Test fraud cases   : {y_test.sum()}")


# ─────────────────────────────────────────────
# 3. BUILD PIPELINES  (imblearn – leak-free)
# ─────────────────────────────────────────────

# --- Pipeline A : Logistic Regression (needs scaling)
pipe_lr = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote',  SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# --- Pipeline B : Random Forest (scale-invariant)
pipe_rf = ImbPipeline(steps=[
    ('smote',  SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
])


# ─────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING WITH GridSearchCV
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# LR param grid
param_grid_lr = {
    'smote__k_neighbors': [3, 5],
    'classifier__C': [0.1, 1.0]
}

# RF param grid
param_grid_rf = {
    'smote__k_neighbors': [3, 5],
    'classifier__max_depth': [10, None]
}

print("\n[TUNING] GridSearchCV – Logistic Regression …")
gs_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=cv,
                     scoring='roc_auc', n_jobs=-1, verbose=0)
gs_lr.fit(X_train, y_train)

print(f"  Best params : {gs_lr.best_params_}")
print(f"  Best CV AUC : {gs_lr.best_score_:.4f}")

print("\n[TUNING] GridSearchCV – Random Forest …")
gs_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cv,
                     scoring='roc_auc', n_jobs=-1, verbose=0)
gs_rf.fit(X_train, y_train)

print(f"  Best params : {gs_rf.best_params_}")
print(f"  Best CV AUC : {gs_rf.best_score_:.4f}")


# ─────────────────────────────────────────────
# 5. EVALUATE ON UNTOUCHED TEST SET
# ─────────────────────────────────────────────

def evaluate(name, model, X_t, y_t):
    y_pred  = model.predict(X_t)
    y_prob  = model.predict_proba(X_t)[:, 1]
    auc     = roc_auc_score(y_t, y_prob)
    ap      = average_precision_score(y_t, y_prob)
    cm      = confusion_matrix(y_t, y_pred)
    report  = classification_report(y_t, y_pred, target_names=["Legit","Fraud"])
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(report)
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    return y_prob, auc, ap, cm

prob_lr, auc_lr, ap_lr, cm_lr = evaluate("Logistic Regression", gs_lr.best_estimator_, X_test, y_test)
prob_rf, auc_rf, ap_rf, cm_rf = evaluate("Random Forest",       gs_rf.best_estimator_, X_test, y_test)


# ─────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Fraud Detection Pipeline – Evaluation Dashboard", fontsize=16, fontweight='bold', y=0.98)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {'lr': '#2196F3', 'rf': '#FF5722'}

# ── (a) Class Distribution
ax0 = fig.add_subplot(gs[0, 0])
counts = y_data.value_counts().sort_index()
bars = ax0.bar(['Legit (0)', 'Fraud (1)'], counts.values,
               color=['#26A69A', '#EF5350'], edgecolor='white', linewidth=1.5)
ax0.set_title('Class Distribution', fontweight='bold')
ax0.set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
             f'{val:,}', ha='center', va='bottom', fontsize=9)

# ── (b) ROC Curves
ax1 = fig.add_subplot(gs[0, 1])
for name, prob, auc, color in [
    ('Logistic Regression', prob_lr, auc_lr, COLORS['lr']),
    ('Random Forest',       prob_rf, auc_rf, COLORS['rf'])
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax1.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
ax1.plot([0,1],[0,1],'k--', lw=1, alpha=0.5, label='Random Classifier')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves', fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# ── (c) Precision-Recall Curves
ax2 = fig.add_subplot(gs[0, 2])
for name, prob, ap, color in [
    ('Logistic Regression', prob_lr, ap_lr, COLORS['lr']),
    ('Random Forest',       prob_rf, ap_rf, COLORS['rf'])
]:
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ax2.plot(rec, prec, color=color, lw=2, label=f'{name} (AP={ap:.3f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── (d) Confusion Matrix – LR
ax3 = fig.add_subplot(gs[1, 0])
im = ax3.imshow(cm_lr, cmap='Blues')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Legit','Fraud']); ax3.set_yticklabels(['Legit','Fraud'])
ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
ax3.set_title('Confusion Matrix – LR', fontweight='bold')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{cm_lr[i,j]:,}', ha='center', va='center',
                 color='white' if cm_lr[i,j] > cm_lr.max()/2 else 'black', fontsize=11)

# ── (e) Confusion Matrix – RF
ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(cm_rf, cmap='Oranges')
ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
ax4.set_xticklabels(['Legit','Fraud']); ax4.set_yticklabels(['Legit','Fraud'])
ax4.set_xlabel('Predicted'); ax4.set_ylabel('Actual')
ax4.set_title('Confusion Matrix – RF', fontweight='bold')
for i in range(2):
    for j in range(2):
        ax4.text(j, i, f'{cm_rf[i,j]:,}', ha='center', va='center',
                 color='white' if cm_rf[i,j] > cm_rf.max()/2 else 'black', fontsize=11)

# ── (f) Metrics Comparison Bar
ax5 = fig.add_subplot(gs[1, 2])
metrics_labels = ['ROC-AUC', 'Avg Precision']
lr_vals = [auc_lr, ap_lr]
rf_vals = [auc_rf, ap_rf]
x = np.arange(len(metrics_labels))
w = 0.3
ax5.bar(x - w/2, lr_vals, width=w, color=COLORS['lr'], label='Logistic Regression')
ax5.bar(x + w/2, rf_vals, width=w, color=COLORS['rf'], label='Random Forest')
ax5.set_ylim(0, 1.1)
ax5.set_xticks(x); ax5.set_xticklabels(metrics_labels)
ax5.set_title('Model Comparison', fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(axis='y', alpha=0.3)
for i, (lv, rv) in enumerate(zip(lr_vals, rf_vals)):
    ax5.text(i - w/2, lv + 0.02, f'{lv:.3f}', ha='center', fontsize=8)
    ax5.text(i + w/2, rv + 0.02, f'{rv:.3f}', ha='center', fontsize=8)

plt.savefig('/mnt/user-data/outputs/fraud_detection_dashboard.png', dpi=150, bbox_inches='tight')
print("\n[DONE] Dashboard saved.")

# ─────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────
winner = "Random Forest" if auc_rf >= auc_lr else "Logistic Regression"
print("\n" + "=" * 60)
print("  PIPELINE SUMMARY")
print("=" * 60)
print(f"  Best Model          : {winner}")
print(f"  LR  ROC-AUC         : {auc_lr:.4f}")
print(f"  RF  ROC-AUC         : {auc_rf:.4f}")
print(f"\n  Zero-Leakage Rules Applied:")
print("  ✔ SMOTE applied INSIDE imblearn pipeline (train fold only)")
print("  ✔ StandardScaler inside pipeline (LR only)")
print("  ✔ Stratified split preserves class ratio")
print("  ✔ Accuracy discarded – using Precision/Recall/ROC-AUC")
print("  ✔ GridSearchCV tunes SMOTE + model params together")
print("=" * 60)
