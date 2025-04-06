import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SpectrogramDataset
from models import GenderCNN

CSV_PATH = "gender_labels.csv"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
TARGET_WIDTH = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SpectrogramDataset(CSV_PATH, target_width=TARGET_WIDTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = GenderCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(dim=1) == y).sum().item()

    acc = correct / len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Accuracy: {acc:.4f}")
