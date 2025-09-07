import sys
import os

# Ensure server root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from app.models.losses import weighted_bce_with_logits, focal_loss_with_logits

def test_weighted_bce():
    logits = torch.tensor([[2.0], [-2.0]])
    targets = torch.tensor([1.0, 0.0])
    loss_unweighted = weighted_bce_with_logits(logits, targets)
    assert loss_unweighted > 0.0
    assert torch.isfinite(loss_unweighted)

    pos_weight = torch.tensor([3.0])
    loss_weighted = weighted_bce_with_logits(logits, targets, pos_weight=pos_weight)
    assert loss_weighted > 0.0
    assert torch.isfinite(loss_weighted)

def test_focal_loss():
    logits = torch.tensor([[10.0], [-10.0]], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    loss_easy = focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0)

    # Hard samples
    logits_hard = torch.tensor([[-5.0], [5.0]], requires_grad=True)
    loss_hard = focal_loss_with_logits(logits_hard, targets, alpha=0.25, gamma=2.0)

    # Focal loss should heavily penalize hard, misclassified samples compared to easy ones
    assert loss_hard > loss_easy
    assert torch.isfinite(loss_easy)
    assert torch.isfinite(loss_hard)

    # Verify backpropagation
    loss_hard.backward()
    assert logits_hard.grad is not None
