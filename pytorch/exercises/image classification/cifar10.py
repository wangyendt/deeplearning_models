#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author:  wangye
@file: cifar10.py 
@time: 2020/03/13
@contact: wangye.hope@gmail.com
@site:  
@software: PyCharm 
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def show_image(img, lbl):
    img = img / 2 + 0.5  # recover image to [0,1]
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.title(' '.join('%5s' % classes[lbl[j]] for j in range(4)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.num_dims(x.size()))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_dims(self, sizes):
        ret = 1
        for size in sizes[1:]:
            ret *= size
        return ret


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = torchvision.datasets.CIFAR10(
        root='../../data', train=True,
        download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True, num_workers=2
    )

    test_set = torchvision.datasets.CIFAR10(
        root='../../data', train=False,
        download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=4, shuffle=False, num_workers=2
    )

    classes = [
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    print(np.shape(train_set.data))
    print(np.shape(test_set.data))
    print(np.shape(train_set.targets))
    print(np.shape(test_set.targets))
    # print(test_set.data[0])
    data_iter = iter(train_loader)
    image, label = data_iter.__next__()
    print(image.size(), label)
    images = torchvision.utils.make_grid(image)
    show_image(images, label)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimzer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            output = net(inputs)
            loss = criterion(output, labels)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
                running_loss = 0.0
    print('Finished Training')

    data_iter = iter(test_loader)
    images, labels = data_iter.__next__()

    show_image(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
