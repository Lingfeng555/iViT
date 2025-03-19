import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST, EMNIST
import pandas as pd
from PIL import Image

class FashionMNISTDataset(Dataset):
    """
    Clase para cargar FashionMNIST utilizando torchvision.
    """
    def __init__(self, root='./data', train=True, transform=None, download=True):
        # Usa transformación a tensor si no se especifica otra
        if transform is None:
            transform = transforms.ToTensor()
        # Carga el dataset FashionMNIST
        self.data = FashionMNIST(root=root, train=train, transform=transform, download=download)
    
    def __len__(self):
        """Retorna el tamaño del dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        # Convert the label (an integer from 0 to 9) into a one-hot encoded vector:
        label_bin = torch.zeros(10, dtype=torch.float)
        label_bin[label] = 1.0
        return image, label_bin
    
class EMNISTDataset(Dataset):
    """
    Carga EMNIST y retorna labels en formato one-hot.
    """
    def __init__(self, root='./data', split='balanced', train=True, transform=None, download=True):
        # Usa transformación a tensor si no se especifica otra
        if transform is None:
            transform = transforms.ToTensor()
        # Carga el dataset EMNIST
        self.data = EMNIST(root=root, split=split, train=train, transform=transform, download=download)
        # Mapear split a número de clases
        num_classes_dict = {
            'byclass': 62,
            'bymerge': 47,
            'balanced': 47,
            'letters': 26,
            'digits': 10,
            'mnist': 10
        }
        self.num_classes = num_classes_dict[split.lower()]
    
    def __len__(self):
        """Retorna el tamaño del dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retorna la imagen y el label one-hot correspondiente al índice idx."""
        image, label = self.data[idx]
        # Convierte el label a vector one-hot
        label_bin = torch.zeros(self.num_classes, dtype=torch.float)
        label_bin[label] = 1.0
        return image, label_bin

    def decode_label(self, label_tensor):
        """
        Devuelve el índice de la etiqueta a partir de un tensor one-hot.
        """
        return torch.argmax(label_tensor).item()
    
if __name__ == '__main__':

    # Crear el dataset
    dataset = EMNISTDataset()
    print(len(dataset))

    # Probar con un DataLoader
    dataloader = DataLoader(dataset, batch_size=900, shuffle=True)

    for images, labels in dataloader:
        print(images.shape, labels.shape)
        print(labels)
        break