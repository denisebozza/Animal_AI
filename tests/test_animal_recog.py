import os, shutil, random, glob
import pytest
from unittest.mock import patch
from PIL import Image
import numpy as np

from src.animal_recogn import eda, training_saving, evaluating, inferencing

RESULTS: str = "./results"
DATASET_FOLDER: str = "./data/animals"
MODEL_PATH_SAVE: str = "./results/animal_model.pth"

IMAGE_PATH: str = "./data/animals/inference"
INFERENCE_PATH: str = "./results/inference"

CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra"
]

def create_dummy_dataset(base_path=DATASET_FOLDER):
    for cls in CLASS_NAMES:
        train_path = os.path.join(base_path, "train", cls)
        test_path  = os.path.join(base_path, "test", cls)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # RGB dummy image (128x128)
        img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        img.save(os.path.join(train_path, f"{cls}_1.jpg"))
        img.save(os.path.join(test_path,  f"{cls}_1.jpg"))

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    create_dummy_dataset()
    yield
    shutil.rmtree("./results/", ignore_errors=True)
    shutil.rmtree("./data/", ignore_errors=True)
    shutil.rmtree("./persistent_data/", ignore_errors=True)

@patch("src.animal_recogn.download")
def test_eda(mock_download):
    mock_download.return_value = None
    eda(download_dataset=True)

    assert os.path.exists(os.path.join(RESULTS, "class_distribution.png"))
    assert os.path.exists(os.path.join(RESULTS, "samples_images.png"))

@patch("src.animal_recogn.download")
def test_training_saving(mock_download):
    mock_download.return_value = None
    training_saving(download_dataset=True, epochs=1, save_path=MODEL_PATH_SAVE, dataset_path=DATASET_FOLDER)

    assert os.path.exists(MODEL_PATH_SAVE)
    assert os.path.getsize(MODEL_PATH_SAVE) > 0

@patch("src.animal_recogn.download")
def test_evaluation(mock_download):
    mock_download.return_value = None
    training_saving(download_dataset=True, epochs=1, save_path=MODEL_PATH_SAVE, dataset_path=DATASET_FOLDER)

    evaluating(
        download_dataset=False,
        model_path=MODEL_PATH_SAVE,
        dataset_path=DATASET_FOLDER,
        path_output=RESULTS,
        verbose=True
    )

    assert os.path.exists(os.path.join(RESULTS, "confusion_matrix.png"))

def get_random_images(root_path: str, num: int = 1):
    image_files = glob.glob(os.path.join(root_path, "**", "*.jpg"), recursive=True)
    if not image_files:
        raise FileNotFoundError(f"No image files found in {root_path}")

    os.makedirs(IMAGE_PATH, exist_ok=True)
    for _ in range(num):
        shutil.copy2(random.choice(image_files), IMAGE_PATH)

@patch("src.animal_recogn.download")
def test_inferencing(mock_download):
    mock_download.return_value = None
    training_saving(download_dataset=True, epochs=1, save_path=MODEL_PATH_SAVE, dataset_path=DATASET_FOLDER)

    get_random_images(os.path.join(DATASET_FOLDER, "test"))
    inferencing(
        image_path=IMAGE_PATH,
        download_dataset=False,
        model_path=MODEL_PATH_SAVE,
        path_output=INFERENCE_PATH
    )

    assert os.path.exists(INFERENCE_PATH)
    pred_classes = os.listdir(INFERENCE_PATH)
    assert len(pred_classes) > 0
    assert any(os.listdir(os.path.join(INFERENCE_PATH, c)) for c in pred_classes)