"""
This is a script for training mnist model.
"""

from __future__ import print_function
import os
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import time
import torch.nn.utils
from torch.autograd import Variable


def dataset():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=50, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=50, shuffle=True)
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


def _train(model, loss_fn, optimizer, data_loader, epochs):
    total_samples = 0
    total_time = 0

    for epoch in range(epochs):
        for step, (x, target) in enumerate(data_loader, 0):
            start_time = time.time()

            output = model(x)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()

            end_time = time.time()
            total_time += end_time - start_time
            total_samples += len(x)

            print('Epoch {} Step {}: speed = {}.'.format(epoch, step, total_samples / total_time))


def _test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    device = torch.device("cpu")
    train_loader, test_loader = dataset()
    model = Net().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    _train(model=model, loss_fn=loss_fn, optimizer=optimizer, data_loader=train_loader, epochs=1)
    _test(model, device, test_loader, loss_fn)
    dummy_input = Variable(torch.randn(1, 1, 28, 28))
    dummy_input = dummy_input.to(device)
    save_path = os.path.join(os.path.dirname(__file__), 'model', 'mnist.onnx')
    dir_name = os.path.dirname(save_path)
    os.makedirs(dir_name, exist_ok=True)
    torch.onnx.export(model, dummy_input, save_path, verbose=True, keep_initializers_as_inputs=True)


if __name__ == '__main__':
    main()
