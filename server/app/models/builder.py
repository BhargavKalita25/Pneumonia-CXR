import torchvision
from torch import nn

def build_model(name: str = "densenet121", num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    """Build a PyTorch classification backbone with customized classifier head."""
    name = name.lower()
    weights = "DEFAULT" if pretrained else None

    if name.startswith("resnet"):
        if not hasattr(torchvision.models, name):
            raise ValueError(f"Unknown ResNet model '{name}'. Available: resnet18, resnet34, resnet50, etc.")
        m = getattr(torchvision.models, name)(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name.startswith("densenet"):
        if not hasattr(torchvision.models, name):
            raise ValueError(f"Unknown DenseNet model '{name}'. Available: densenet121, densenet169, densenet201, etc.")
        m = getattr(torchvision.models, name)(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    elif name.startswith("efficientnet"):
        if not hasattr(torchvision.models, name):
            raise ValueError(f"Unknown EfficientNet model '{name}'. Available: efficientnet_b0, efficientnet_b2, etc.")
        m = getattr(torchvision.models, name)(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported architecture '{name}'. Supported families: 'resnet*', 'densenet*', 'efficientnet*'"
        )

    return m
