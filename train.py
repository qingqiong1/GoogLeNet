from torchvision import datasets
from torchvision import transforms
import torch.utils.data
import copy
import torch
from torch import nn
import time
from model import googlenet
from model import Inception
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from PIL import Image
def train_val_process():
    ROOT_TRAIN = r'data\train'
    ROOT_TEST = r'data\test'
    mean=[0.1620883,0.15117277,0.13852146]
    std =[0.05796373,0.05223298,0.04782545]
    normalize = transforms.Normalize(mean,std)
    r_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),normalize])

    warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

    train_data = datasets.ImageFolder(ROOT_TRAIN, transform=r_transform)
    test_data = datasets.ImageFolder(ROOT_TEST, transform=r_transform)

    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [int(0.8 * len(train_data)),len(train_data) - int(0.8 * len(train_data))])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
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
        print("第{}轮训练开始".format(epoch + 1))
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

def test_print(model, test_loader,classes):
    '''
    use:用于打印预测值与真实值
    :param model: 模型
    :param test_loader: 测试数据
    :param classes: 每一类的名称
    :return:没有返回值
    '''
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load('best_model.pt'))
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            print("epoch:",i,"--","预测值：",classes[predicted.item()],"---","真实值：",classes[labels.item()],"---","是否正确:",predicted.item()==labels.item())


def recog_image(image_path,model,classes):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load('best_model.pt'))

    image = Image.open(image_path, 'r')
    mean = [0.1620883, 0.15117277, 0.13852146]
    std = [0.05796373, 0.05223298, 0.04782545]
    normalize = transforms.Normalize(mean, std)
    r_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = r_transform(image).unsqueeze(0)
    with torch.no_grad():
        images = image.to(device)
        outputs =model(images)
        predicted = torch.argmax(outputs, dim=1)
        print("预测值：", classes[predicted.item()])


if __name__ == '__main__':
    model = googlenet(Inception)
    #train_loader, val_loader, test_loader = train_val_process()
    # #train_process = train(model, train_loader, val_loader, epochs=20, lr=0.001)
    # #test(model, test_loader)
    # test_print(model,test_loader,["猫","狗"])
    # #matplot_acc_loss(train_process)
    recog_image('1.jpg',model,["猫","狗"])
