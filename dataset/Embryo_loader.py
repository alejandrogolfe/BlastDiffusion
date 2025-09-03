import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List
import torchvision
from torchvision import transforms
import difflib

class Embryo_loader(Dataset):
    """
    The SiCAPv2 class represents a Histopathology dataset of pixel-level annotations of prostate patches with different Gleason Grades.
     It loads images from a specified directory and their corresponding labels from a CSV file. The class supports optional
     transformations to preprocess the images and provides methods for accessing the dataset in a way suitable for training machine learning models.
    """

    def __init__(self, csv_file: str, root_dir: str, im_channels: int):
        dataframe = pd.read_csv(csv_file)
        self.image_names = dataframe["IMAGE_NAME"].astype(str) + ".png"
        self.labels = dataframe["blastocist"]
        self.root_dir = root_dir  # Store the directory of images
        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * im_channels, std=[0.5] * im_channels)
        ])
        self.image_list: List[torch.Tensor] = []  # List to store images
        self.labels_list: List[torch.Tensor] = []  # List to store labels
        self.image_paths = []

        # List all files in the root directory
        self.all_images = os.listdir(self.root_dir)
        print('Loading images to memory...')

        # Load images and their labels
        for position, element in enumerate(tqdm(self.image_names, desc="Loading")):
            # Find the most similar image in the root directory
            closest_image = self.get_closest_image(element, self.all_images)

            if closest_image:
                img_path = os.path.join(self.root_dir, closest_image)  # Create full image path
                try:
                    image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
                    image = self.transform(image)  # Apply transform if provided
                    y_label = torch.tensor(self.labels[position], dtype=torch.long)  # Convert label to tensor
                    self.image_list.append(image)  # Store the image in the list
                    self.labels_list.append(y_label)  # Store the label in the list
                    self.image_paths.append(closest_image)  # Store the image path
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")  # Handle image loading errors

    def get_closest_image(self, image_name, all_images):
        """Find the most similar image name from the list of available images."""
        # Remove the file extension and compare
        image_name_without_ext = image_name.split('.')[0]
        close_matches = difflib.get_close_matches(image_name_without_ext, all_images, n=1, cutoff=0.8)

        if close_matches:
            # Ensure the match has the '.png' extension
            return close_matches[0] if close_matches[0].endswith(".png") else None
        return None

    def change_transform(self, transform: torchvision.transforms.Compose) -> None:
        """
        Change the transformation applied to the images in the dataset.

        Args:
            transform (torchvision.transforms.Compose): New transformation to apply to the images.
        """
        self.transform = transform  # Update the transformation

    @staticmethod
    def transform_labels(dataframe: pd.DataFrame, dictionary_classes: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform labels in the DataFrame into a format that can be used by the model.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image names and labels.
            dictionary_classes (dict): A dictionary mapping text labels to numerical indices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the following elements:

                - X (np.ndarray): A 0D tensor that represents the names of the image names in string format.
                - y (np.ndarray): A 0D tensor that represents the transformed labels.
        
        """

        X, y = [], []  # Lists to store image names and labels

        for i, row in dataframe.iterrows():

            label = None  # Initialize label variable

            # Identify which label column has a 1
            for class_name, class_index in dictionary_classes.items():
                if row.get(class_name, 0) == 1:  # Check if the class column has a 1
                    label = class_index
                    break

            if label is None:
                continue
            X.append(row[0])  # Store image name
            y.append(label)

        return np.array(X), np.array(y)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return an element from the dataset given its index.

        Args:
            index (int): Index of the element to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the following elements:

                - image (torch.Tensor): A 3D tensor of shape (img_nchan, img_w, img_h) that represents the transformed image, where:
                    
                    - **img_nchan**: Number of input channels (e.g., 3 for RGB images).
                    - **img_w**: The width of the image in pixels.
                    - **img_h**: The height of the image in pixels.
                - label (torch.Tensor): A 0D tensor that represents the corresponding label.
        """
        image = self.image_list[index]  # Retrieve the image
        label = self.labels_list[index]  # Retrieve the corresponding label

          # Apply the transformation to the image if available
        return {
            "data": image,
            "class": label,
        }

class Validation_loader(Dataset):
    def __init__(self, val_data, directory_inception_features=None):
        """
        Inicializa el dataset personalizado.

        Args:
            image_names (list of str): Listado de nombres de imágenes (con extensión).
            labels (list of int): Etiquetas correspondientes a las imágenes.
            directory_inception_features (str): Directorio base donde se encuentran los archivos .pt.
        """
        self.data = val_data
        self.image_names = val_data.image_paths
        self.class_labels = torch.tensor(val_data.labels_list, dtype=torch.long)
        self.directory_inception_features=directory_inception_features
        self.inception_features=[]
        # Cargar los tensores .pt en memoria
        for i, img_name in enumerate(self.image_names):
            self.inception_features.append(torch.load(os.path.join(self.directory_inception_features, os.path.splitext(img_name)[0] + ".pt")))
        self.inception_features = torch.stack(self.inception_features) if isinstance(self.inception_features[0], torch.Tensor) else self.inception_features

    def __len__(self):
        """
        Devuelve el número de muestras en el dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Devuelve una muestra en formato dict.

        Args:
            idx (int): Índice de la muestra.

        Returns:
            dict: Contiene las características y la etiqueta de la muestra.
        """
        return {
            "inception_features": self.inception_features[idx],
            "class": self.class_labels[idx],
        }


