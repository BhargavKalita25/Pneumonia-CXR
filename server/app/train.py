import os
import sys
import json

# Ensure server root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim

from app.config import TrainConfig, load_profile
from app.data.dataset import CXRFolder, class_from_path
from app.models.builder import build_model
from app.models.losses import weighted_bce_with_logits, focal_loss_with_logits
from app.utils.seed import set_seed
from app.utils.metrics import compute_all_metrics
from app.utils.plots import plot_training_curves

def get_pos_weight(ds: CXRFolder) -> torch.Tensor:
    """
    Extract class labels from file paths in O(1) memory and milliseconds,
    avoiding the massive I/O overhead of decoding every image in the dataset.
    """
    labels = [class_from_path(p) for p in ds.items]
    pos = sum(labels)
    neg = len(labels) - pos
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, scaler, device, loss_name, pos_weight):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    # Modern AMP context manager
    use_cuda = device == "cuda"
    for x, y, _, _ in tqdm(loader, leave=False, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_cuda and scaler.is_enabled():
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                if loss_name == "focal":
                    loss = focal_loss_with_logits(logits, y)
                else:
                    loss = weighted_bce_with_logits(logits, y, pos_weight.to(device) if pos_weight is not None else None)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            if loss_name == "focal":
                loss = focal_loss_with_logits(logits, y)
            else:
                loss = weighted_bce_with_logits(logits, y, pos_weight.to(device) if pos_weight is not None else None)
            loss.backward()
            optimizer.step()

        prob = torch.sigmoid(logits).squeeze(1)
        pred = (prob >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)

    return (loss_sum / total), (correct / total)

@torch.no_grad()
def validate(model, loader, device, loss_name="bce", pos_weight=None):
    model.eval()
    y_true, y_prob, val_loss_sum, total = [], [], 0.0, 0

    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1).cpu()
        y_true += y.cpu().numpy().tolist()
        y_prob += prob.numpy().tolist()

        if loss_name == "focal":
            loss = focal_loss_with_logits(logits, y)
        else:
            loss = weighted_bce_with_logits(logits, y, pos_weight.to(device) if pos_weight is not None else None)

        val_loss_sum += loss.item() * y.size(0)
        total += y.size(0)

    metrics = compute_all_metrics(y_true, y_prob)
    metrics["val_loss"] = float(val_loss_sum / max(total, 1))
    return metrics

def main():
    cfg = load_profile(TrainConfig())
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing PneumoniaCXR Training on {device.upper()} (Profile: {cfg.profile})")

    # Datasets & Loaders
    train_ds = CXRFolder(cfg.data_root, "train", cfg.img_size, train=True)
    val_ds = CXRFolder(cfg.data_root, "val", cfg.img_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Model construction
    model = build_model(cfg.model_name, num_classes=1, pretrained=True).to(device)

    # Backbone warmup: freeze feature extraction layers
    if cfg.freeze_backbone_epochs > 0:
        for n, p in model.named_parameters():
            if not ("classifier" in n or "fc" in n):
                p.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.scheduler == "cosine"
        else optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    )

    # Modern AMP scaler
    scaler = torch.amp.GradScaler(device="cuda", enabled=(cfg.amp and device == "cuda"))

    pos_weight = None if cfg.class_weight == 0 else torch.tensor([cfg.class_weight])
    if cfg.class_weight == 0:
        pos_weight = get_pos_weight(train_ds)
        print(f"Computed positive class weight: {pos_weight.item():.3f}")

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_auc, best_path = -1.0, os.path.join(cfg.output_dir, "best.ckpt")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        # Unfreeze backbone and re-link scheduler so LR schedule stays active
        if epoch == cfg.freeze_backbone_epochs and cfg.freeze_backbone_epochs > 0:
            print(f"Unfreezing backbone at epoch {epoch + 1}")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            remaining_epochs = max(cfg.epochs - epoch, 1)
            scheduler = (
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
                if cfg.scheduler == "cosine"
                else optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
            )

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg.loss, pos_weight
        )

        metrics = validate(model, val_loader, device, cfg.loss, pos_weight)
        val_loss, val_auc, val_acc = metrics["val_loss"], metrics["roc_auc"], metrics["accuracy"]

        if cfg.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_auc)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1:02d}/{cfg.epochs:02d} | "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.3f}, Acc: {val_acc:.3f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, best_path)
            with open(os.path.join(cfg.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f" [BEST] New best model saved with Val ROC-AUC: {best_auc:.3f}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stop_patience:
            print(f"[STOP] Early stopping triggered at epoch {epoch+1}")
            break

    plot_training_curves(hist, cfg.output_dir)
    print(f"[DONE] Training completed. Best checkpoint saved at: {best_path}")

if __name__ == "__main__":
    main()
