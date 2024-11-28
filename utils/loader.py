import os
import urllib.request
import zipfile
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch


def get_env_vars() -> None:
    env = "local"
    data_path = os.path.join(os.curdir, "data")
    return env, data_path


def download_dataset(data_path: str) -> None:
    data_path = data_path
    url = "https://www.kaggle.com/api/v1/datasets/download/zeyad1mashhour/driver-inattention-detection-dataset"
    filename = "dataset.zip"

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    file_path = os.path.join(data_path, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    else:
        print(f"{filename} already exists.")

    # 압축 해제
    if filename.endswith(".zip"):
        extract_path = os.path.join(data_path, os.path.splitext(filename)[0])
        if not os.path.exists(extract_path):
            print(f"Unzipping {filename}...")
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Unzipped to: {extract_path}")
        else:
            print(f"{filename} is already unzipped at {extract_path}")
    else:
        print("Unzipping skipped.")


class Annotation:
    def __init__(
        self,
        image_name: str,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        class_id: int,
    ):
        self.image_name = image_name
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.class_id = class_id

    def __repr__(self):
        return f"Annotation(image_name={self.image_name}, x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max}, class_id={self.class_id})"


def get_annotations(file_path: str) -> list[Annotation]:
    annotations = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name = parts[0]
            try:
                bbox = list(map(int, parts[1].split(",")))
                if len(bbox) != 5:
                    raise ValueError
                annotation = Annotation(image_name, *bbox)
                annotations.append(annotation)
            except ValueError:
                continue

    return annotations


class CustomDataset(Dataset):
    def __init__(self, X: list, y: list, mode="train"):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform: transforms.Compose

        # mode에 따라 다른 transform 적용
        if mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.transform(self.X[idx])
        y = self.y[idx]
        return (x, y)


def get_dataset(
    annotations: list[Annotation], data_path: str, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    target_size = (224, 224)

    images = []
    labels = []

    for annotation in annotations:
        image_path = os.path.join(data_path, annotation.image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        images.append(image)
        labels.append(annotation.class_id)

    images = np.array(images)
    labels = np.array(labels)

    return CustomDataset(images, labels, mode)


def get_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    env, data_path = get_env_vars()

    print(f"Running in {env} environment, data path is {data_path}")

    download_dataset(data_path=data_path)

    train_data_path = os.path.join(data_path, "dataset", "train")
    test_data_path = os.path.join(data_path, "dataset", "test")
    valid_data_path = os.path.join(data_path, "dataset", "valid")

    train_annotations = get_annotations(
        os.path.join(train_data_path, "_annotations.txt")
    )
    test_annotations = get_annotations(os.path.join(test_data_path, "_annotations.txt"))
    valid_annotations = get_annotations(
        os.path.join(valid_data_path, "_annotations.txt")
    )

    train_dataset = get_dataset(train_annotations, train_data_path, "train")
    test_dataset = get_dataset(test_annotations, test_data_path, "test")
    valid_dataset = get_dataset(valid_annotations, valid_data_path, "valid")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    return train_loader, test_loader, valid_loader


if __name__ == "__main__":
    train_loader, test_loader, valid_loader = get_loaders()
    print(train_loader)
    print(test_loader)
    print(valid_loader)
