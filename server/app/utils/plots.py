import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(hist: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(hist["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(hist["val_loss"], label="Val Loss", color="darkorange", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss.png"), dpi=200)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(7, 5))
    plt.plot(hist["train_acc"], label="Train Accuracy", color="royalblue", lw=2)
    plt.plot(hist["val_acc"], label="Val Accuracy", color="forestgreen", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc.png"), dpi=200)
    plt.close()

def plot_roc_pr(metrics: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    fpr, tpr = metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"]
    prec, rec = metrics["pr_curve"]["precision"], metrics["pr_curve"]["recall"]

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc.png"), dpi=200)
    plt.close()

    # PR
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="purple", lw=2, label=f"PR (AUC = {metrics['pr_auc']:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr.png"), dpi=200)
    plt.close()

def plot_calibration(y_true, y_prob, outdir: str, bins: int = 10):
    os.makedirs(outdir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    qs = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(y_prob, qs) - 1, 0, bins - 1)

    means = [y_prob[idx == i].mean() if np.any(idx == i) else np.nan for i in range(bins)]
    fracs = [y_true[idx == i].mean() if np.any(idx == i) else np.nan for i in range(bins)]

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.scatter(means, fracs, color="crimson", s=40, zorder=5)
    plt.plot(means, fracs, color="crimson", lw=1.5, label="Model Reliability")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Positive Fraction")
    plt.title("Reliability / Calibration Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "calibration.png"), dpi=200)
    plt.close()
