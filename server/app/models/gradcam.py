import torch
from torch import nn
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        modules = dict([*self.model.named_modules()])
        if target_layer not in modules:
            raise KeyError(
                f"Target layer '{target_layer}' not found in model. Available modules: {list(modules.keys())[:15]}..."
            )

        layer = modules[target_layer]
        h_fwd = layer.register_forward_hook(self._fwd)
        h_bwd = layer.register_full_backward_hook(self._bwd)
        self.hook_handles.extend([h_fwd, h_bwd])

    def _fwd(self, m, inp, out):
        self.activations = out.detach()

    def _bwd(self, m, gin, gout):
        self.gradients = gout[0].detach()

    def remove(self):
        """Clean up registered hooks to prevent memory leaks."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def __call__(self, x: torch.Tensor):
        # Enable gradients explicitly so Grad-CAM works even if called inside torch.no_grad()
        with torch.enable_grad():
            self.model.zero_grad()
            logits = self.model(x)
            score = logits[:, 0].sum()
            score.backward()

        grads, acts = self.gradients, self.activations
        if grads is None or acts is None:
            raise RuntimeError("Failed to capture feature gradients or activations for Grad-CAM.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        prob = torch.sigmoid(logits).detach().cpu().item()
        return cam.detach().cpu().numpy()[0, 0], prob

def overlay_cam(rgb_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap onto RGB image.
    Automatically handles spatial resizing if CAM dimensions do not match the image.
    """
    h, w = rgb_img.shape[:2]

    # Resize CAM to image resolution if dimensions differ
    if cam.shape[:2] != (h, w):
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    heat = (cam * 255.0).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1.0 - alpha) * rgb_img).clip(0, 255).astype(np.uint8)
    return overlay
