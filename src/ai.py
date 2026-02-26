import os, shutil, time
from pathlib import Path

from dotenv import load_dotenv
import kagglehub

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

RESULTS = "./results/"


class AI:
    @staticmethod
    def get_set_loader_ofDataSetImg(transform, path: str, batch: int, shuffle: bool):
        sets = ImageFolder(root=path, transform=transform)
        loader = DataLoader(sets, batch_size=batch, shuffle=shuffle)
        return sets, loader


class SimpleCNN(nn.Module):
    """
    Resolution-agnostic CNN via AdaptiveAvgPool2d, so no hardcoded Linear input size.
    """
    def __init__(self, total_class_num: int, in_channels: int = 3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, total_class_num),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class AnimalRecognAI(AI):
    X, Y = 128, 128
#miao 
    CLASS_NAMES = [
        "Beetle", "Butterfly", "Cat", "Cow", "Dog",
        "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
        "Mouse", "Panda", "Spider", "Tiger", "Zebra"
    ]
    CLASS_MAP = {i: name for i, name in enumerate(CLASS_NAMES)}

    def __init__(self, model_path: str = None, rgb: bool = True):
        self.in_channels = 3 if rgb else 1
        self.model = SimpleCNN(len(self.CLASS_MAP), in_channels=self.in_channels)
        if model_path is not None:
            self._load_model(model_path)
        self.transform = self._get_transforms(rgb=rgb)

    def _get_transforms(self, rgb: bool = True):
        t = [
            transforms.Resize((self.X, self.Y)),
            transforms.ToTensor(),
        ]
        if not rgb:
            t.insert(0, transforms.Grayscale(num_output_channels=1))
        return transforms.Compose(t)

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        obj = torch.load(model_path, map_location="cpu")
        if isinstance(obj, dict):
            self.model.load_state_dict(obj)
            print(f"State dict loaded from {model_path}")
        else:
            self.model = obj
            print(f"Full model loaded from {model_path}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model, path)
        print(f"Model saved to {path}")

    def save_state_dict(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model state_dict saved to {path}")

    @staticmethod
    def download_dataset(path: str, kaggle_dataset: str = "utkarshsaxenadn/animal-image-classification-dataset"):
        load_dotenv()
        try:
            user = kagglehub.whoami()
            print(f"✅ Autenticated as: {user}")
        except Exception as e:
            print(f"❌ Autentication Error: {e}")
            raise SystemExit(1) from e

        if path is not None:
            os.makedirs(path, exist_ok=True)

        try:
            cache_path = kagglehub.dataset_download(kaggle_dataset, force_download=True)
            print(f"🔄 Dataset scaricato nella cache: {cache_path}")

            for item in os.listdir(cache_path):
                src = os.path.join(cache_path, item)
                dst = os.path.join(path, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

            print(f"✅ Dataset finale in: {path}")

            user_name, dataset_name = kaggle_dataset.split('/')
            cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / user_name / dataset_name
            shutil.rmtree(cache_root, ignore_errors=True)
            print(f"🗑️  Pulizia cache: {cache_root}")

        except Exception as e:
            print(f"❌ Errore durante l'operazione: {str(e)}")
            if os.path.exists(path):
                shutil.rmtree(path)
            raise SystemExit(1) from e

    def EDA(
        self, dataset_path: str,
        batch_train: int = 64, batch_test: int = 1000,
        shuffle_train: bool = True, shuffle_test: bool = True,
        path_output: str = RESULTS,
        num_sample_img: int = 10
    ):
        train_set, train_loader = AI.get_set_loader_ofDataSetImg(
            self.transform, os.path.join(dataset_path, "train"), batch_train, shuffle_train
        )
        test_set, _ = AI.get_set_loader_ofDataSetImg(
            self.transform, os.path.join(dataset_path, "test"), batch_test, shuffle_test
        )

        print("EDA Reports:")
        print("- Train data size: ", len(train_set))
        print("- Test data size: ", len(test_set))

        labels = np.array(train_set.targets)
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))

        print("- Class distribution: ", end="")
        for class_idx, count in sorted(class_dist.items()):
            print(f"[{self.CLASS_MAP[int(class_idx)]}: {count}]", end=" ")
        print()

        os.makedirs(path_output, exist_ok=True)

        # Plot class distribution
        plt.figure(figsize=(12, 4))
        x_names = [self.CLASS_MAP[int(k)] for k in class_dist.keys()]
        sns.barplot(x=x_names, y=list(class_dist.values()))
        plt.title("Class distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, "class_distribution.png"))
        plt.close()

        # Sample Images
        examples = enumerate(train_loader)
        _, (examples_data, examples_targets) = next(examples)

        n = min(num_sample_img, len(examples_data))
        cols = 5
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(12, 3 * rows))

        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            img = examples_data[i]
            if img.shape[0] == 1:
                plt.imshow(img[0], cmap="gray", interpolation="none")
            else:
                plt.imshow(img.permute(1, 2, 0))
            plt.title(f"{self.CLASS_MAP[examples_targets[i].item()]}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(path_output, "samples_images.png"))
        plt.close()

    def train(
        self, path_data: str,
        epochs: int = 5, batch_size: int = 64, shuffle: bool = True,
        device: str = None
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        _, train_loader = AI.get_set_loader_ofDataSetImg(
            self.transform, os.path.join(path_data, "train"), batch_size, shuffle
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        start = time.time()
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}")

        print(f"Training time: {time.time()-start:.2f} seconds")

    def evaluate(
        self, path_data: str,
        batch_size: int = 256, shuffle: bool = False, verbose: bool = False,
        path_output: str = RESULTS,
        device: str = None
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        _, test_loader = AI.get_set_loader_ofDataSetImg(
            self.transform, os.path.join(path_data, "test"), batch_size, shuffle
        )

        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                predicted = outputs.argmax(dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total if total > 0 else 0.0

        if verbose:
            print(f"Test accuracy: {accuracy:.4f}")

            target_names = [self.CLASS_MAP[i] for i in range(len(self.CLASS_MAP))]
            print(classification_report(all_labels, all_preds, target_names=target_names))

            conf_matrix = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                conf_matrix, annot=False, fmt='d', cmap='Purples',
                xticklabels=target_names, yticklabels=target_names
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()

            os.makedirs(path_output, exist_ok=True)
            name = os.path.join(path_output, "confusion_matrix.png")
            plt.savefig(name)
            plt.close()
            print(f"Saved confusion matrix to {name}")

        return accuracy

    def inference_singleImg(self, path_img: str, path_output: str, device: str = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        image = read_image(path_img)  # [C,H,W], uint8
        image_tensor = self.transform(to_pil_image(image)).unsqueeze(0).to(device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1).item()
            predicted_class = self.CLASS_MAP[pred]

        save_dir = os.path.join(path_output, predicted_class)
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.basename(path_img)
        save_path = os.path.join(save_dir, filename)

        # save the processed tensor (normalized 0..1) as image
        save_image(image_tensor.squeeze(0).cpu(), save_path)
        print(f"Inference: {predicted_class} -> {save_path}")