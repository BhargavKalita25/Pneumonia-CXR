import os
import sys
import json
import argparse

# Ensure server root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from app.config import TrainConfig, load_profile
from app.data.dataset import CXRFolder
from app.models.builder import build_model
from app.utils.metrics import compute_all_metrics
from app.utils.plots import plot_roc_pr, plot_calibration

def main():
    parser = argparse.ArgumentParser(description="Evaluate PneumoniaCXR Model on Test Set")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation plots and metrics")
    args = parser.parse_args()

    cfg = load_profile(TrainConfig())
    ckpt_path = args.checkpoint or os.path.join(cfg.output_dir, "best.ckpt")
    data_root = args.data_root or cfg.data_root
    out_dir = args.output_dir or cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
        print("Please train the model first by running: npm run train")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating {cfg.model_name} on {device.upper()} from checkpoint: {ckpt_path}")

    test_ds = CXRFolder(data_root, "test", cfg.img_size, train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg.model_name, num_classes=1, pretrained=False).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y, _, _ in test_loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x)).squeeze(1).cpu().numpy().tolist()
            y_prob.extend(prob)
            y_true.extend(y.numpy().tolist())

    metrics = compute_all_metrics(y_true, y_prob)

    metrics_file = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_roc_pr(metrics, out_dir)
    plot_calibration(y_true, y_prob, out_dir)

    print("\nEvaluation Results:")
    print(f"  - ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  - PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"  - Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  - Precision:  {metrics['precision']:.4f}")
    print(f"  - Recall:     {metrics['recall']:.4f}")
    print(f"  - F1-Score:   {metrics['f1']:.4f}")
    print(f"  - Brier:      {metrics['brier']:.4f}")
    print(f"\n[DONE] Metrics and diagnostic curves saved to: {out_dir}")

if __name__ == "__main__":
    main()
