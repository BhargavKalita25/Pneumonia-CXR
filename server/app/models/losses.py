import torch
import torch.nn.functional as F

def weighted_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor = None) -> torch.Tensor:
    """Binary cross entropy with optional positive class weighting."""
    return F.binary_cross_entropy_with_logits(
        logits, targets.unsqueeze(1).float(), pos_weight=pos_weight
    )

def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Numerically stable Focal Loss for binary classification.
    Correctly weights positive class by alpha and negative class by (1 - alpha).
    """
    t = targets.unsqueeze(1).float()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p = torch.sigmoid(logits)
    pt = t * p + (1.0 - t) * (1.0 - p)
    alpha_t = t * alpha + (1.0 - t) * (1.0 - alpha)
    focal = alpha_t * ((1.0 - pt) ** gamma) * bce
    return focal.mean()
