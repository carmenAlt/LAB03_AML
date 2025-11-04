import torch
from tqdm import tqdm
from models.model import CustomNet
from torch import nn, optim


def train(epoch, model, train_loader, criterion, optimizer, device):
    """
    Esegue un'epoca di addestramento.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f"✅ Train Epoch: {epoch} | Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.2f}%")

    return train_loss, train_accuracy

if __name__ == "__main__":
    print("🚀 Starting training script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Device in use: {device}")

    # 1️⃣ Scarica o trova il dataset
    print("⬇️ Downloading / checking dataset...")
    data_path = download_and_extract_dataset()
    print(f"📁 Dataset ready at: {data_path}")

    # 2️⃣ Prepara i DataLoader
    print("🔧 Preparing data loaders...")
    train_loader, val_loader, num_classes = prepare_data(data_dir=data_path)
    print(f"✅ Data prepared! Num classes: {num_classes}")
    print(f"Train batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    # 3️⃣ Inizializza modello, loss e ottimizzatore
    print("🧠 Initializing model...")
    model = CustomNet(num_classes=num_classes).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🎯 Starting training loop...")
    for epoch in range(1, 4):  # 3 epoche giusto per test
        train(epoch, model, train_loader, criterion, optimizer, device)

    torch.save(model.state_dict(), "model.pth")
    print("💾 Model saved as model.pth ✅")
    print("🏁 Training completed successfully!")
