import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import SpectrogramDataset
from transformer_model import AudioTransformer, AudioTransformerV2
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


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
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

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

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    return history


def plot_results(history, task, model_type):
    os.makedirs("plots", exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title(f"{task.capitalize()} - {model_type.upper()} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plots/{task}_{model_type}_loss.png")

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.title(f"{task.capitalize()} - {model_type.upper()} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"plots/{task}_{model_type}_accuracy.png")

    print("\nPlots saved in /plots/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='labels.csv', help='Path to CSV with labels and npy paths')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, choices=['transformer', 'vit'], default='transformer')
    parser.add_argument('--task', type=str, choices=['gender', 'age'], default='gender')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension for transformer')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size for ViT-style transformer')

    args = parser.parse_args()

    print(f"Model: {args.model}, Task: {args.task}, Device: {args.device}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"Embed dim: {args.embed_dim}, Num heads: {args.num_heads}, Num layers: {args.num_layers}")
    print(f"Patch size: {args.patch_size}")

    dataset = SpectrogramDataset(args.csv, target_width=128, task=args.task)
    input_shape = (128, dataset.target_width)

    labels = dataset.df[dataset.label_column].values
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    num_classes = 2 if args.task == 'gender' else 7
    
    if args.model == 'transformer':
        model = AudioTransformer(
            num_classes=num_classes, 
            input_shape=input_shape,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=0.1
        )
    elif args.model == 'vit':
        model = AudioTransformerV2(
            num_classes=num_classes, 
            input_shape=input_shape,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            patch_size=args.patch_size,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model.to(args.device)
    
    criterion = nn.CrossEntropyLoss()    
    lr = args.lr / 5.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = train(model, train_loader, val_loader, optimizer, criterion, args.device, args.epochs)

    plot_results(history, task=args.task, model_type=args.model)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(args.device), y.to(args.device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))


if __name__ == '__main__':
    main()