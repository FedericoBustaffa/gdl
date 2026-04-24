import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split
from tqdm import trange


class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.01) -> None:
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            layers.append(nn.Linear(fan_in, fan_out))

            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        # layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        self.sgd = optim.SGD(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, tr_loader, val_loader, max_iter: int = 200) -> None:
        self.history = {"tr": [], "val": []}
        for epoch in trange(max_iter):
            self.network.train()
            epoch_loss = 0.0
            for x, y in tr_loader:
                self.sgd.zero_grad()
                pred = self.network(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.sgd.step()

                epoch_loss += loss.item()

            self.history["tr"].append(epoch_loss / len(tr_loader))

            self.network.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    pred = self.network(x)
                    loss = self.loss_fn(pred, y)
                    epoch_val_loss += loss.item()

            self.history["val"].append(epoch_val_loss / len(val_loader))

    def predict(self, loader) -> torch.Tensor:
        self.network.eval()
        preds = []

        with torch.no_grad():
            for x, _ in loader:
                logits = self.network(x)
                probs = F.softmax(logits, dim=1)
                pred = torch.masked.argmax(probs, dim=1)

                preds.append(pred)

        return torch.cat(preds)


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, y)

    tr_split = int(0.7 * len(dataset))
    val_split = len(dataset) - tr_split

    tr, val = random_split(dataset, [tr_split, val_split])

    tr_loader = dataloader.DataLoader(tr, 32, shuffle=True)
    val_loader = dataloader.DataLoader(val, 32)

    network = NeuralNetwork([64, 64, 10])
    network.fit(tr_loader, val_loader, max_iter=100)

    y_true = []
    y_pred = []

    network.network.eval()

    with torch.no_grad():
        for x, y in val_loader:
            logits = network.network(x)
            preds = torch.argmax(logits, dim=1)

            y_true.append(y)
            y_pred.append(preds)

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    print(f"accuracy: {accuracy:.2f}")

    plt.figure(figsize=(6, 4), dpi=150)
    plt.title("Training Curve")
    plt.plot(network.history["tr"], label="training")
    plt.plot(network.history["val"], label="training")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
