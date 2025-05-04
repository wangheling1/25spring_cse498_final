# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:18:16 2025

@author: 81265
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

# === CONFIG ===
MAT_PATH = "small_subset.mat"
EPOCHS = 5
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === DATA LOADING ===
with h5py.File(MAT_PATH, "r") as f:
    data = np.array(f["Data"]).astype(np.float32)  # shape (N, H, W)
    mask = np.array(f["SegmentBitmap"]).astype(np.float32)

# Add channel dimension
data = data[:, np.newaxis, :, :]         # (N, 1, H, W)
mask = mask[:, np.newaxis, :, :]         # (N, 1, H, W)

# === DATASET ===
class SegmentationDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])

# === DATA SPLIT ===
x_temp, x_test, y_temp, y_test = train_test_split(data, mask, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

train_loader = DataLoader(SegmentationDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SegmentationDataset(x_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(SegmentationDataset(x_test, y_test), batch_size=BATCH_SIZE)

# === MODEL ===
class SimpleSegNet(nn.Module):
    def __init__(self):
        super(SimpleSegNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # H/2
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)  # H/4
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# === TRAINING LOOP ===
model = SimpleSegNet().to(DEVICE)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# === TESTING ===
model.eval()
test_loss = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        test_loss += loss.item()
print(f"\nTest Loss: {test_loss:.4f}")
