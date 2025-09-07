from io import BytesIO
from typing import Any
import numpy as np
from PIL import Image
import pydicom

def _to_scalar(val: Any) -> float:
    """Safely convert DICOM tag value to float, handling MultiValue collections."""
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        return float(list(val)[0])
    return float(val)

def dicom_to_png_bytes(dcm_bytes: bytes) -> bytes:
    """
    Convert raw DICOM bytes to standard RGB PNG bytes.
    Handles MultiValue window levels, Modality LUT scaling, and MONOCHROME1 inversion.
    """
    ds = pydicom.dcmread(BytesIO(dcm_bytes), force=True)
    arr = ds.pixel_array.astype(np.float32)

    # 1. Apply Modality LUT (Rescale Slope & Intercept)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        slope = _to_scalar(ds.RescaleSlope)
        intercept = _to_scalar(ds.RescaleIntercept)
        arr = arr * slope + intercept

    # 2. Photometric Interpretation inversion for MONOCHROME1
    photo = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    if photo == "MONOCHROME1":
        arr = np.amax(arr) - arr

    # 3. VOI LUT Window Center & Width
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        try:
            wc = _to_scalar(ds.WindowCenter)
            ww = _to_scalar(ds.WindowWidth)
            if ww > 0:
                arr = np.clip(arr, wc - ww / 2.0, wc + ww / 2.0)
        except (ValueError, TypeError, IndexError):
            pass

    # 4. Normalize to [0, 255] uint8
    arr_min = float(np.amin(arr))
    arr_max = float(np.amax(arr))
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        arr = np.zeros_like(arr)

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")

    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
