
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataset import ExpertDataset


DATA_PATH = "data/expert/ppo_expert.h5"
MODEL_PATH = "experiments/bc_baseline/bc_best.pt"

EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
SEED = 42


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, 3)

    def forward(self, x):
        features = self.encoder(x)
        raw = self.actor(features)

        steering = torch.tanh(raw[:, 0:1])
        gas = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])

        return torch.cat([steering, gas, brake], dim=1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with h5py.File(DATA_PATH, "r") as f:
        episode_names = sorted(f.keys())

    rng = np.random.default_rng(SEED)
    rng.shuffle(episode_names)

    split = max(1, int(len(episode_names) * 0.8))

    train_names = episode_names[:split]
    val_names = episode_names[split:]

    if not val_names:
        raise ValueError("At least two episodes are required.")

    train_dataset = ExpertDataset(DATA_PATH, train_names)
    val_dataset = ExpertDataset(DATA_PATH, val_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = BCPolicy().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("Device:", device)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, actions in train_loader:
            images = images.to(device)
            actions = actions.to(device)

            predictions = model(images)
            loss = criterion(predictions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, actions in val_loader:
                images = images.to(device)
                actions = actions.to(device)

                predictions = model(images)
                loss = criterion(predictions, actions)

                val_loss += loss.item() * len(images)

        val_loss /= len(val_dataset)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print("Best model saved.")

    print("Training completed.")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()
