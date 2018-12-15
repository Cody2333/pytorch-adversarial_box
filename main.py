#coding=utf-8

# Copyright 2017 - 2018 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CNN on cifar data using pytorch and adversarial training
"""
from __future__ import print_function

import os
import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
from adversarialbox.attacks import FGSMAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
train_data = torchvision.datasets.CIFAR10(
 './cifar-adv-pytorch/data', train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.CIFAR10(
 './cifar-adv-pytorch/data', train=False, transform=torchvision.transforms.ToTensor()
)
# print("train_data:", train_data.train_data.size)
# print("train_labels:", train_data.train_labels.size)
# print("test_data:", test_data.test_data.size)


#批大小
batch_size=128
#训练的批次数
epochs=20

delay =10
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 5, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )


    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return torch.nn.functional.log_softmax(out, dim=1)



def main():

    # 自适应使用GPU还是CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()


    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size)

    adversary = FGSMAttack(epsilon=0.2)

    for epoch in range(epochs):
        for t, (x, y) in enumerate(train_loader):

            x_var, y_var = to_var(x), to_var(y.long())
            loss = criterion(model(x_var), y_var)

            # adversarial training
            if epoch + 1 > delay:
                # use predicted label to prevent label leaking
                y_pred = pred_batch(x, model)
                x_adv = adv_train(x, y_pred, model, criterion, adversary)
                x_adv_var = to_var(x_adv)
                loss_adv = criterion(model(x_adv_var), y_var)
                loss = (loss + loss_adv) / 2

            if (t + 1) % 10 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 每跑完一次epoch测试一下准确率 进入测试模式 禁止梯度传递
        with torch.no_grad():
            correct = 0
            total = 0
            sum_val_loss = 0
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                val_loss = criterion(outputs, labels)
                sum_val_loss += val_loss.item()
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('epoch=%d accuracy=%.02f%% val_loss=%.02f%' % (epoch + 1, (100 * correct / total), sum_val_loss))
            sum_val_loss = 0.0
            

    torch.save(model.state_dict(), './cifar-adv-pytorch/net.pth')




if __name__ == '__main__':
    main()

