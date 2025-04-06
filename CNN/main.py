import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import SpectrogramDataset
from models import GenderCNN, GenderMLP


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='gender_labels.csv', help='Path to CSV with labels and npy paths')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, choices=['cnn', 'mlp'], default='cnn')
    parser.add_argument('--task', type=str, choices=['gender', 'age'], default='gender')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # === Load dataset ===
    dataset = SpectrogramDataset(args.csv, target_width=128, task=args.task)
    input_shape = (128, dataset.target_width)

    # === Train/Val split ===
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # === Model ===
    num_classes = 2 if args.task == 'gender' else 9
    if args.model == 'cnn':
        model = GenderCNN(num_classes=num_classes, input_shape=input_shape)
    else:
        model = GenderMLP(input_shape=input_shape, num_classes=num_classes)
    model.to(args.device)

    # === Training ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, train_loader, val_loader, optimizer, criterion, args.device, args.epochs)


if __name__ == '__main__':
    main()
