import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_model import LetterCNN

def train():
    device = torch.device('cpu')
    print(f"Training on: {device}")
    print("RAM tip: close browser tabs and other apps to free memory!\n")

    # Transform defined first
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset loading uses transform
    print("Loading EMNIST letters dataset...")
    train_data = datasets.EMNIST(
        root=r'C:\emnist_data', split='letters',
        train=True, download=True, transform=transform
    )
    test_data = datasets.EMNIST(
        root=r'C:\emnist_data', split='letters',
        train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples:     {len(test_data)}\n")

    model     = LetterCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = (labels - 1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

            if i % 200 == 0:
                acc = 100 * correct / total if total > 0 else 0
                print(f"Epoch {epoch+1}/5 — Batch {i}/{len(train_loader)} — "
                      f"Loss: {running_loss/(i+1):.4f} — Acc: {acc:.1f}%")

        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = (labels - 1).to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"\nEpoch {epoch+1} complete — Validation accuracy: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'cnn_weights.pth')
            print(f"New best! Saved at {val_acc:.2f}%\n")

    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
    print("Weights saved to cnn_weights.pth")

if __name__ == "__main__":
    train()