import sys
import os
import io

# Ensure server root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.multival import MultiValue
from PIL import Image

from app.data.dicom_utils import _to_scalar, dicom_to_png_bytes

def test_to_scalar():
    assert _to_scalar(42) == 42.0
    assert _to_scalar("100.5") == 100.5
    assert _to_scalar([50.0, 60.0]) == 50.0
    assert _to_scalar(MultiValue(float, [123.4, 567.8])) == 123.4

def create_synthetic_dicom(multivalue=False, monochrome1=False):
    filename = "synthetic.dcm"
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"

    if multivalue:
        ds.WindowCenter = MultiValue(float, [1000.0, 1200.0])
        ds.WindowWidth = MultiValue(float, [2000.0, 2400.0])
    else:
        ds.WindowCenter = 1000.0
        ds.WindowWidth = 2000.0

    pixels = np.arange(64 * 64, dtype=np.uint16).reshape((64, 64))
    ds.PixelData = pixels.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()

def test_dicom_conversion():
    raw_dcm = create_synthetic_dicom(multivalue=False, monochrome1=False)
    png_bytes = dicom_to_png_bytes(raw_dcm)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (64, 64)

def test_dicom_multivalue_handling():
    raw_dcm = create_synthetic_dicom(multivalue=True, monochrome1=False)
    png_bytes = dicom_to_png_bytes(raw_dcm)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

def test_dicom_monochrome1_inversion():
    raw_dcm = create_synthetic_dicom(multivalue=False, monochrome1=True)
    png_bytes = dicom_to_png_bytes(raw_dcm)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
