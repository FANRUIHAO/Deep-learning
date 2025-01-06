import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import csv
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import time


class CovidDatatest(Dataset):
    def __init__(self, file_path, mode):
        with open(file_path, "r") as f:
            ori_data = list(csv.reader(f))
            csv_data = np.array(ori_data[1:])[:, 1:].astype(float)

        if mode == "train":
            indices = [i for i in range(len(csv_data)) if i % 5 != 0]
            self.y = torch.tensor(csv_data[indices, -1])
        elif mode == "val":
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]  # fix: validation set indices
            self.y = torch.tensor(csv_data[indices, -1])
        else:
            indices = [i for i in range(len(csv_data))]

        data = torch.tensor(csv_data[indices, :-1])
        self.data = (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True)
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode != "test":
            return self.data[idx].float(), self.y[idx].float()
        else:
            return self.data[idx].float()

    def __len__(self):
        return len(self.data)


class MyModel(nn.Module):
    def __init__(self, inDim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(inDim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        if len(x.size()) > 1:
            return x.squeeze(1)

        return x


def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):
    model = model.to(device)

    plt_train_loss = []
    plt_val_loss = []
    main_val_loss = 999999999

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()

        model.train()
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_bat_loss = loss(pred, target)
            train_bat_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += train_bat_loss.cpu().item()

        plt_train_loss.append(train_loss / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_bat_loss = loss(pred, target)
                val_loss += val_bat_loss.cpu().item()
        plt_val_loss.append(val_loss / len(val_loader.dataset))
        if val_loss < main_val_loss:
            torch.save(model, save_path)
            main_val_loss = val_loss

        print('[%03d/%03d] %2.2f sec(s) TrainLoss : %.6f | valLoss: %.6f' % \
              (epoch + 1, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1]))

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss图")
    plt.legend(["train", "val"])
    plt.show()


device = "cuda" if torch.cuda.is_available() else "cpu"

train_file = "covid.train.csv"

test_file = "covid.test.csv"

train_dataset = CovidDatatest(train_file, "train")
val_dataset = CovidDatatest(train_file, "val")
test_dataset = CovidDatatest(test_file, "test")

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print(device)

config = {
    "lr": 0.01,
    "epochs": 20,
    "momentum": 0.9,
    "save_path": "model_save/best_model.pth"
}

model = MyModel(inDim=93).to(device)
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])