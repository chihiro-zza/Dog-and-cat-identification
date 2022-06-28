import matplotlib.pyplot as plt

import re

model = ['Alexnet', 'GoogLeNet', 'moblieNet', 'pretrained_module', 'ResNet']
dict = {}
for net in model:
    pos = './' + net + "/test.txt"
    with open(pos, "r+", encoding='utf-8') as f:
        loss = []
        acc = []
        for line in f.readlines():
            i = line.strip()[:-1]
            pattern = re.compile(r'[0-9]+')
            nums = re.findall(pattern, i)
            loss.append(float(nums[0] + nums[1]) / 10000.0)
            acc.append(float(nums[-1]) / 100.0)

        dict[net] = {'loss': loss, 'acc': acc}

for net in model:
    plt.plot(dict[net]['loss'], label=net)
    plt.title('Validation Loss')
    plt.legend()

plt.show()

for net in model:

    plt.plot(dict[net]['acc'], label=net)
    plt.title('Validation Accuracy')
    plt.legend()

plt.show()