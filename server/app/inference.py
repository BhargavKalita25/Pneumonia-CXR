import os
import threading
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn

from app.models.builder import build_model
from app.models.gradcam import GradCAM, overlay_cam

class Predictor:
    def __init__(self, arch: str, ckpt_path: str, img_size: int = 224, device: str = "cpu"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found at '{ckpt_path}'. Train the model first via: npm run train"
            )

        self.device = device
        self.lock = threading.Lock()
        self.model = build_model(arch, num_classes=1, pretrained=False).to(device)

        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from '{ckpt_path}': {e}")

        self.model.eval()

        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Select the most suitable convolutional feature map layer for Grad-CAM
        if "densenet" in arch.lower():
            self.target_layer = "features.denseblock4"
        elif "resnet" in arch.lower():
            self.target_layer = "layer4"
        else:
            # Automatic discovery of the last Conv2d layer
            last_conv_name = None
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv_name = name
            if not last_conv_name:
                raise ValueError("Could not find a Conv2d layer in the model for Grad-CAM.")
            self.target_layer = last_conv_name

        self.cam = GradCAM(self.model, target_layer=self.target_layer)

    def predict_with_cam(self, pil_img: Image.Image, out_path: str):
        """Thread-safe inference and Grad-CAM overlay generation."""
        rgb = pil_img.convert("RGB")
        orig_w, orig_h = rgb.size
        x = self.tfm(rgb).unsqueeze(0).to(self.device)

        # Thread lock prevents concurrent state collision during forward/backward hook execution
        with self.lock:
            try:
                cam, prob = self.cam(x)
            except Exception as e:
                raise RuntimeError(f"Grad-CAM inference failed: {e}")

        # Resize CAM to match original image spatial dimensions
        cam_img = Image.fromarray((cam * 255.0).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.BILINEAR)
        cam_resized = np.array(cam_img, dtype=np.float32) / 255.0

        # Create overlay and save
        overlay = overlay_cam(np.array(rgb), cam_resized, alpha=0.45)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        Image.fromarray(overlay).save(out_path)

        label = "Pneumonia" if prob >= 0.5 else "Normal"
        confidence = prob if prob >= 0.5 else (1.0 - prob)

        return label, float(confidence)
