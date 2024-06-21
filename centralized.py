# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 00:33:23 2024

@author: Nemo
"""

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset


num_clients=2
BATCH_SIZE=32

DEVICE = torch.device("cuda")
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, trainloader, epochs: int, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in tqdm(trainloader, "Training"):
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total * 100
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy






def load_datasets(partition_id):
    fds = FederatedDataset(dataset='cifar10', partitioners={"train": num_clients }, shuffle=True, seed=42)

    def apply_transform(batch):
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    partition = fds.load_partition(partition_id)
    partition = partition.train_test_split(train_size=.8, seed=42)
    partition = partition.with_transform(apply_transform)
    trainloader = DataLoader(partition["train"], shuffle=True, batch_size=BATCH_SIZE)
    valloader = DataLoader(partition["test"], batch_size=BATCH_SIZE)
        
    return trainloader, valloader




