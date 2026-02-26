import os

try:
    from ai import AnimalRecognAI
except ImportError:
    from src.ai import AnimalRecognAI

RESULTS: str = "./results"
DATASET_FOLDER: str = "./data/animals"
MODEL_PATH: str = "./persistent_data/animal_model.pth"
MODEL_PATH_SAVE: str = "./results/animal_model.pth"
IMAGE_PATH: str = "./data/animals/inference"
INFERENCE_PATH: str = "./results/inference"


def download():
    AnimalRecognAI.download_dataset(path=DATASET_FOLDER)


def eda(download_dataset: bool = False):
    if download_dataset:
        download()
    ai = AnimalRecognAI()
    ai.EDA(DATASET_FOLDER, path_output=RESULTS)


def training_saving(
    download_dataset: bool = False,
    epochs: int = 5,
    save_path: str = MODEL_PATH_SAVE,
    dataset_path: str = DATASET_FOLDER,
):
    if download_dataset:
        download()
    ai = AnimalRecognAI()
    ai.train(path_data=dataset_path, epochs=epochs)
    ai.save_state_dict(path=save_path)
    return ai


def evaluating(
    download_dataset: bool = False,
    model_path: str = MODEL_PATH,
    dataset_path: str = DATASET_FOLDER,
    path_output: str = RESULTS,
    verbose: bool = True,
):
    if download_dataset:
        download()

    if not os.path.exists(model_path):
        ai = training_saving(epochs=5, dataset_path=dataset_path, save_path=model_path)
    else:
        ai = AnimalRecognAI(model_path=model_path)

    ai.evaluate(path_data=dataset_path, path_output=path_output, verbose=verbose)


def inferencing(
    image_path: str,
    download_dataset: bool = False,
    model_path: str = MODEL_PATH,
    path_output: str = INFERENCE_PATH,
):
    if download_dataset:
        download()
    if not os.path.exists(model_path):
        raise FileNotFoundError("Necessario un modello pre-addestrato esistente")

    ai = AnimalRecognAI(model_path=model_path)

    for filename in os.listdir(image_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            ai.inference_singleImg(
                path_img=os.path.join(image_path, filename),
                path_output=path_output
            )


if __name__ == "__main__":
    download()
    eda()
    evaluating(model_path=MODEL_PATH_SAVE)
    inferencing(IMAGE_PATH, model_path=MODEL_PATH_SAVE)