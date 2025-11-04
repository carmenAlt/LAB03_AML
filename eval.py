import torch
from models.model import CustomNet
from torch import nn, optim
from utils.download_dataset import download_and_extract_dataset
from dataset.dataset import prepare_data

def evaluate():
    print("🚀 Starting evaluation script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Device in use: {device}")

    # 1️⃣ Carica il dataset
    print("⬇️ Downloading / checking dataset...")
    data_path = download_and_extract_dataset()
    print(f"📁 Dataset ready at: {data_path}")

    _, val_loader, num_classes = prepare_data(data_dir=data_path)
    print(f"✅ Validation set loaded with {num_classes} classes")

    # 2️⃣ Carica il modello addestrato
    print("🧠 Loading trained model...")
    model = CustomNet().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")

    criterion = nn.CrossEntropyLoss()

    # 3️⃣ Esegui validazione
    print("🔍 Running evaluation...")
    validate(model, val_loader, criterion, device)
    print("🏁 Evaluation completed successfully!")

def validate(model, val_loader, criterion, device):
    """
    Esegue la validazione del modello su un DataLoader.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total

    print(f"🧪 Validation Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
    return val_loss, val_accuracy

