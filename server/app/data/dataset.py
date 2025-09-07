import os
import glob
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def default_transforms(img_size: int, train: bool):
    if train:
        aug = [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
        ]
    else:
        aug = [transforms.Resize((img_size, img_size))]

    aug += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(aug)

def class_from_path(path: str) -> int:
    return 1 if "PNEUMONIA" in path.upper() else 0

def patient_id_from_filename(path: str) -> str:
    """
    Extract a unique patient identifier from Kaggle CXR filenames.
    Prevents false clustering of NORMAL2-IM-* into synthetic ID '2'.
    """
    stem = os.path.splitext(os.path.basename(path))[0]

    # Pattern 1: person{ID}_{bacteria|virus}_...
    m_person = re.search(r"(person\d+)", stem, re.IGNORECASE)
    if m_person:
        return m_person.group(1).lower()

    # Pattern 2: (NORMAL|NORMAL2|BACTERIA)-IM-{ID}-{IMAGE} or IM-{ID}-{IMAGE}
    m_im = re.search(r"IM-(\d+)", stem, re.IGNORECASE)
    if m_im:
        return f"patient_{m_im.group(1)}"

    # Fallback to extracting the longest numeric run, or stem
    digits = re.findall(r"\d+", stem)
    if digits:
        return max(digits, key=len)
    return stem

class CXRFolder(Dataset):
    def __init__(self, root: str, split: str, img_size: int, train: bool):
        self.root = root
        self.split = split
        pattern = os.path.join(root, split, "*", "*")
        all_files = sorted(glob.glob(pattern))

        # Filter supported extensions
        valid_exts = {".png", ".jpg", ".jpeg"}
        self.items = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]

        if not self.items:
            raise RuntimeError(f"No valid CXR images found in {root}/{split}")

        self.tfm = default_transforms(img_size, train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        try:
            with Image.open(p) as img:
                rgb_img = img.convert("RGB")
                tensor = self.tfm(rgb_img)
        except Exception as e:
            raise RuntimeError(f"Failed to read image at {p}: {e}")

        y = class_from_path(p)
        pid = patient_id_from_filename(p)
        return tensor, torch.tensor(y, dtype=torch.float32), p, pid
