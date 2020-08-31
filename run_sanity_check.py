import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from GaborNet.GaborLayerNew import GaborConv2d
from dataset import DogsCatsDataset


class GaborNN(nn.Module):
    def __init__(self, device):
        super(GaborNN, self).__init__()
        self.g1 = GaborConv2d(3, 32, kernel_size=(15, 15), stride=1)
        self.c1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.g1(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=2)
        x = F.max_pool2d(F.leaky_relu(self.c2(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = x.view(-1, 128 * 7 * 7)
        x = F.leaky_relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = self.fc3(x)
        return x


def main():
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = DogsCatsDataset(root_dir=os.path.join('data', 'train'),
                                transform=transform)
    test_set = DogsCatsDataset(root_dir=os.path.join('data', 'val'),
                               transform=transform)

    train = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    test = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = GaborNN(device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters())

    one_layer_gnet_acc_train = []
    one_layer_gnet_acc_test = []

    for epoch in range(50):

        running_loss = 0.0
        correct = 0
        net.train()
        for i, data in enumerate(train):
            # get the inputs
            inputs, labels = data['image'], data['target']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            pred = outputs.max(1, keepdim=True)[1].to('cpu')
            correct += pred.eq(labels.view_as(pred)).sum().item()

            # print statistics
            running_loss += loss.item()
        print('[%d] train_acc: %.3f train_loss: %.3f' % (
            epoch + 1, correct / len(train_set), running_loss / len(train_set)))
        one_layer_gnet_acc_train.append(correct / len(train_set))

        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(test, 0):
                # get the inputs
                inputs, labels = data['image'], data['target']

                # forward + backward + optimize
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                pred = outputs.max(1, keepdim=True)[1].to('cpu')
                correct += pred.eq(labels.view_as(pred)).sum().item()
                running_loss += loss.item()
        print('[%d] test_acc: %.3f test_loss: %.3f' % (
            epoch + 1, correct / len(test_set), running_loss / len(test_set)))
        one_layer_gnet_acc_test.append(correct / len(test_set))

    print('Finished Training')

    result_dict = {'train_acc': one_layer_gnet_acc_train[-1],
                   'test_acc': one_layer_gnet_acc_test[-1]}
    with open('metrics.json', 'w+') as outfile:
        json.dump(result_dict, outfile)


if __name__ == '__main__':
    main()
