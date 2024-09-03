import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split


class FaceDataset(Dataset):
    """
    A custom dataset class for face images.

    Parameters:
    - image_paths (list): A list of file paths to the face images.
    - labels (list): A list of labels corresponding to the face images.
    - transform (callable, optional): A function/transform to apply to the images.

    Returns:
    - tuple: Returns a tuple containing the transformed image and its corresponding label.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def dynamic_transforms(transform_config):
    """Create a Compose object from a transform config list."""
    transform_list = []
    for transform in transform_config:
        for key, value in transform.items():
            if key == "Normalize":
                transform_list.append(getattr(transforms, key)(**value))
            else:
                transform_list.append(getattr(transforms, key)(**value))
    return transforms.Compose(transform_list)


def load_data(config):
    """
    Load data from the given configuration.

    Parameters:
    - config (dict): A dictionary containing the necessary configuration settings.

    Returns:
    - tuple: Returns training and validation dataloaders along with the training labels.
    """
    labels = np.loadtxt(config["data"]["labels_file"], dtype=int)
    image_paths = [os.path.join(config["data"]["train_path"], f"{i+1:06d}.jpg") for i in range(len(labels))]

    # Split the data into training and validation sets in memory
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels)

    # Load transforms from config
    train_transform = dynamic_transforms(config["transforms"]["train"])
    val_transform = dynamic_transforms(config["transforms"]["val"])

    # Create datasets
    train_dataset = FaceDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = FaceDataset(val_paths, val_labels, transform=val_transform)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size_train"], shuffle=True, num_workers=config["data"]["num_workers"]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["training"]["batch_size_val"], shuffle=False, num_workers=config["data"]["num_workers"]
    )

    return train_loader, val_loader, train_labels
