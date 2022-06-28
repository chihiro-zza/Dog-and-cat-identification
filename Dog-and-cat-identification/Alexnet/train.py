import os
import sys
import json
import torch.nn.functional as F
import torch
import torch.nn as nn
from PIL.Image import Image
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(size=(244, 244)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset_train = datasets.ImageFolder('D:/WorkSpace/image/kaggle_Dog&Cat/train', transform)

    dataset_test = datasets.ImageFolder('D:/WorkSpace/new_image/kaggle_Dog&Cat/validation', transform)

    # 导入数据

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    modellr = 1e-4
    model_save_path = 'model.pth'
    # 实例化模型并且移动到GPU
    if os.path.exists(model_save_path):
        print('-------------load the model-----------------')
        model = torch.load(model_save_path)
    else:
        model = AlexNet(num_classes=1, init_weights=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=modellr)

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        modellrnew = modellr * (0.1 ** (epoch // 5))
        print("lr:", modellrnew)
        for param_group in optimizer.param_groups:
            param_group['lr'] = modellrnew

    # 定义训练过程
    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            output = model(data)

            loss = F.binary_cross_entropy(output, target)

            loss.backward()

            optimizer.step()

            if (batch_idx + 1) % 69 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                           100. * (batch_idx + 1) / len(train_loader), loss.item()))

    # 定义测试过程
    def val(model, device, test_loader):
        # 评估模式。在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
        model.eval()

        test_loss = 0

        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).float().unsqueeze(1)

                output = model(data)
                test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
                pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)
                correct += pred.eq(target.long()).sum().item()

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            with open("test.txt", "a") as f:
                f.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

    def predicted():
        class_names = ['cat', 'dog']

        model = torch.load(model_save_path)
        model.eval()

        image_PIL = Image.open('D:/WorkSpace/image/test1/1.jpg')
        image_PIL.show()
        image_tensor = transform(image_PIL)
        # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor.unsqueeze_(0)
        # 没有这句话会报错
        image_tensor = image_tensor.to(device)

        out = model(image_tensor)
        pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in out]).to(DEVICE)
        print(class_names[pred])

    # 训练
    for epoch in range(1, 10 + 1):
        adjust_learning_rate(optimizer, epoch)
        train(model, device, train_loader, optimizer, epoch)
        val(model, device, test_loader)

    torch.save(model, model_save_path)

    predicted()



if __name__ == '__main__':
    main()
