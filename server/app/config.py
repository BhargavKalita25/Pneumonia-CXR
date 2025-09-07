import os
import yaml
from dataclasses import dataclass, field

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@dataclass
class TrainConfig:
    data_root: str = "data/kaggle_chest_xray"
    output_dir: str = os.path.join(BASE_DIR, "outputs")
    model_name: str = "densenet121"
    img_size: int = 224
    batch_size: int = 64
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    loss: str = "bce"  # "bce" or "focal"
    class_weight: float = 0.0
    amp: bool = True
    seed: int = 42
    num_workers: int = 4
    freeze_backbone_epochs: int = 1
    early_stop_patience: int = 4
    profile: str = "gpu_t4"

@dataclass
class ServeConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 15
    allowed_ext: tuple = (".png", ".jpg", ".jpeg", ".dcm")
    model_registry: dict = field(default_factory=lambda: {
        "default": {
            "path": os.path.join(BASE_DIR, "outputs", "best.ckpt"),
            "arch": "densenet121",
            "img_size": 224
        }
    })

def load_profile(cfg: TrainConfig) -> TrainConfig:
    profile_path = os.path.join(os.path.dirname(__file__), "profiles", f"{cfg.profile}.yaml")
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            prof = yaml.safe_load(f) or {}
        for k, v in prof.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg
