# 导入库
import os
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torchvision import models

BATCH_SIZE = 16
# 迭代次数
EPOCHS = 10
# 采用cpu还是gpu进行计算
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# 使用VGG特征
model = models.vgg16(pretrained=True)
vgg_feature = model.features  # 训练的时候忘记设置vgg模式为eval()，也就是说vgg的参数在训练的时候会发生改变

'''使用Resnet特征
resnet = models.resnet34(pretrained=True)
modules = list(resnet.children())[:-2]  # delete the last fc layer.
res_feature = nn.Sequential(*modules).eval()'''


class MyNet(nn.Module):
    def __init__(self, vgg_feature):
        super(MyNet, self).__init__()
        self.vgg_feature = vgg_feature

        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.vgg_feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


model_save_path = 'model.pth'
# 实例化模型并且移动到GPU
if os.path.exists(model_save_path):
    print('-------------load the model-----------------')
    model = torch.load(model_save_path)
else:
    model = MyNet(vgg_feature).to(DEVICE)

# 选择简单暴力的Adam优化器，学习率调低

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

        if (batch_idx + 1) % 99 == 0:
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


# 训练
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, DEVICE, train_loader, optimizer, epoch)
    val(model, DEVICE, test_loader)

torch.save(model, model_save_path)



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
    image_tensor = image_tensor.to(DEVICE)

    out = model(image_tensor)
    pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in out]).to(DEVICE)
    print(class_names[pred])



predicted()