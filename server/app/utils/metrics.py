import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)

def compute_all_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)

    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) > 1

    if has_both_classes:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
        precs, recs, pr_thr = precision_recall_curve(y_true, y_prob)

        # Sensitivity at 90% Specificity
        spec = 1.0 - fpr
        sens_at_90spec = float(tpr[spec >= 0.90].max()) if np.any(spec >= 0.90) else None

        # Specificity at 90% Sensitivity
        spec_at_90sens = float(spec[tpr >= 0.90].max()) if np.any(tpr >= 0.90) else None

        roc_dict = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thr": [float(x) for x in roc_thr],
        }
        pr_dict = {
            "precision": [float(x) for x in precs],
            "recall": [float(x) for x in recs],
            "thr": [float(x) for x in pr_thr],
        }
    else:
        roc_auc = 0.5
        pr_auc = float(y_true.mean()) if len(y_true) > 0 else 0.0
        sens_at_90spec = None
        spec_at_90sens = None
        roc_dict = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thr": [1.0, 0.0]}
        pr_dict = {"precision": [pr_auc], "recall": [1.0], "thr": [0.0]}

    acc = float((y_pred == y_true).mean())
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    calib = float(brier_score_loss(y_true, y_prob))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "sensitivity_at_90_specificity": sens_at_90spec,
        "specificity_at_90_sensitivity": spec_at_90sens,
        "roc_curve": roc_dict,
        "pr_curve": pr_dict,
        "brier": calib,
    }
