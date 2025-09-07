import sys
import os

# Ensure server root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import torch
from app.data.dataset import default_transforms, patient_id_from_filename, class_from_path

def test_transform_output_shape():
    tfm_eval = default_transforms(224, train=False)
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    tensor = tfm_eval(img)
    assert tensor.shape == (3, 224, 224)
    assert torch.isfinite(tensor).all()

    tfm_train = default_transforms(224, train=True)
    tensor_train = tfm_train(img)
    assert tensor_train.shape == (3, 224, 224)
    assert torch.isfinite(tensor_train).all()

def test_patient_id_extraction():
    # Verify that NORMAL2-IM-... does not extract synthetic ID '2'
    assert patient_id_from_filename("data/NORMAL2-IM-0262-0001.jpeg") == "patient_0262"
    assert patient_id_from_filename("data/IM-0115-0001.jpeg") == "patient_0115"
    assert patient_id_from_filename("data/person100_bacteria_475.jpeg") == "person100"
    assert patient_id_from_filename("data/person19_virus_50.jpeg") == "person19"

def test_class_from_path():
    assert class_from_path("data/train/PNEUMONIA/person1_bacteria_1.jpeg") == 1
    assert class_from_path("data/train/NORMAL/IM-0001-0001.jpeg") == 0
    assert class_from_path("data/test/PNEUMONIA/test_sample.png") == 1
