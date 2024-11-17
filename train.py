from torchvision import datasets
from torchvision import transforms
import torch.utils.data
import numpy as np
import os
import copy
import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
import time
from model import googlenet
from model import Inception
import pandas as pd
import matplotlib.pyplot as plt


def train_val_process():
    dataset = datasets.FashionMNIST("./data", train=True, download=True,
                                    transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,), (0.5,))]))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)),
                                                                                       int(0.1 * len(dataset)),
                                                                                       len(dataset) - int(
                                                                                           0.9 * len(dataset))])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, epochs, lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 初始化参数
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    best_acc = 0.0
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        train_num = 0
        val_num = 0
        # 训练过程
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_num += images.size(0)
            train_loss += loss.item() * images.size(0)
            train_acc += torch.sum(torch.argmax(outputs, dim=1) == labels.view(-1)).item()
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_acc / train_num)
        print("第{}次训练，训练损失为：{:.4f}，训练准确率为：{:.2f}%"
              .format(epoch + 1, train_loss_all[-1], train_acc_all[-1] * 100))

        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_num += images.size(0)
            val_loss += loss.item() * images.size(0)
            val_acc += torch.sum(torch.argmax(outputs, dim=1) == labels.view(-1)).item()
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_acc / val_num)
        print("第{}训练，验证损失为：{:.4f}，验证准确率为：{:.2f}%".format(epoch + 1, val_loss_all[-1],
                                                                       val_acc_all[-1] * 100))
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("{} Time use: {:.0f}m{:.0f}s".format(epoch + 1, time_use // 60, time_use % 60))
    torch.save(best_model, 'best_model.pt')

    train_process = pd.DataFrame(data={
        "epoch": range(1, epochs + 1),
        "train_loss": train_loss_all,
        "train_acc": train_acc_all,
        "val_loss": val_loss_all,
        "val_acc": val_acc_all
    })
    # val_process = pd.DataFrame(data={
    #     "epoch": range(1, epoch + 1),
    #     "train_loss": train_loss_all,
    #     "train_acc": train_acc_all,
    #     "val_loss": val_loss_all,
    #     "val_acc": val_acc_all
    # })
    return train_process


def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load('best_model.pt'))
    loss = 0
    acc = 0
    test_num = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_num += images.size(0)
            loss += criterion(outputs, labels).item() * images.size(0)
            acc += torch.sum(torch.argmax(outputs, dim=1) == labels.view(-1)).item()
        print("训练损失为：{:.4f}，训练准确率为：{:.2f}%".format(loss / test_num, acc * 100 / test_num))


def matplot_acc_loss(process):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(process['epoch'], process["train_loss"], 'ro-', label="train_loss")
    plt.plot(process['epoch'], process["val_loss"], 'bs-', label="val_loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(process['epoch'], process["train_acc"], 'ro-', label="train_acc")
    plt.plot(process['epoch'], process["val_acc"], 'bs-', label="val_acc")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    model = googlenet(Inception)
    train_loader, val_loader, test_loader = train_val_process()
    train_process = train(model, train_loader, val_loader, epochs=1, lr=0.001)
    test(model, test_loader)
    matplot_acc_loss(train_process)
