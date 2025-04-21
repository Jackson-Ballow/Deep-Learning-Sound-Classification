import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import SpectrogramDataset
from xgboost_model import AudioXGBoost
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train(model, train_loader, val_loader, epochs):
    return model.fit(train_loader, val_loader, num_boost_round=epochs)


def plot_results(history, task, model_type):
    os.makedirs("plots", exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title(f"{task.capitalize()} - {model_type.upper()} Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plots/{task}_{model_type}_loss.png")

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.title(f"{task.capitalize()} - {model_type.upper()} Accuracy")
    plt.xlabel("Round")
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
    parser.add_argument('--model', type=str, choices=['xgboost'], default='xgboost')
    parser.add_argument('--task', type=str, choices=['gender', 'age'], default='gender')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Model: {args.model}, Task: {args.task}, Device: {args.device}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")

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
    model = AudioXGBoost(num_classes=num_classes)
    history = train(model, train_loader, val_loader, args.epochs)

    plot_results(history, task=args.task, model_type=args.model)

    all_preds = []
    all_labels = []

    all_preds = model.predict(val_loader)
    _, all_labels = model._extract_features(val_loader)

    print("\nFinal Classification Report:")
    print(accuracy_score(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))


if __name__ == '__main__':
    main()