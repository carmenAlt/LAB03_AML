import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np


def prepare_data(data_dir="data/tiny-imagenet-200", batch_size=32, visualize=False):
    """
    Prepara Tiny ImageNet: applica le trasformazioni, crea DataLoader e (opzionalmente) visualizza esempi.
    """

    # 1️⃣ Trasformazioni standard
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2️⃣ Percorsi dataset
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "val")  # Tiny ImageNet usa "val" come test

    # 3️⃣ Controlli
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ Train folder not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ Validation folder not found: {test_dir}")

    # 4️⃣ Dataset + DataLoader
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    num_samples = len(train_dataset)

    print(f"✅ Dataset ready!")
    print(f"   Number of classes: {num_classes}")
    print(f"   Number of training samples: {num_samples}")

    # 5️⃣ Visualizza alcune immagini (opzionale)
    if visualize:
        visualize_samples(dataloader_train)

    # 6️⃣ Ritorna i DataLoader e le info principali
    return dataloader_train, dataloader_test, num_classes


def denormalize(image):
    """Rimuove la normalizzazione per la visualizzazione."""
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def visualize_samples(dataloader, n_samples=10):
    """Mostra immagini di esempio dal DataLoader."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    sampled_classes = []
    count = 0

    for inputs, labels in dataloader:
        for img, label in zip(inputs, labels):
            if label.item() not in sampled_classes:
                ax = axes[count // 5, count % 5]
                ax.imshow(denormalize(img))
                ax.set_title(f'Class: {label.item()}')
                ax.axis('off')
                sampled_classes.append(label.item())
                count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break

    plt.show()
