import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms,datasets


# PyTorch Dataset class for loading CIFAR10
#
# Torchvision already has an implementation for CIFAR, but we'll do it ourselves
# for educational purposes. This one can be easily adapted to custom datasets.
#
class CIFAR10(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []
        name_part = "data_batch" if train else "test_batch"

        # load image data, possibly from many files
        for filename in os.listdir(path):
            if name_part in filename:
                with open(os.path.join(path, filename), 'rb') as fp:
                    filedata = pickle.load(fp, encoding='latin1')
                    self.data.append(filedata['data'])
                    self.targets.extend(filedata['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

        # load "meta data", i.e., class names
        meta_path = os.path.join(path, 'batches.meta')
        with open(meta_path, 'rb') as fp:
            metadata = pickle.load(fp, encoding='latin1')
            self.classes = metadata['label_names']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)

        return image, target


# Define a simple CNN model, adapted from this tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
#
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('WARNING: No GPU available, using CPU')
        device = torch.device('cpu')

    # Define data loaders
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = CIFAR10(args.data_path, train=True, transform=transform)
    trainset = datasets.CIFAR10(root='./dataset', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    # testset = CIFAR10(args.data_path, train=False, transform=transform)
    testset = datasets.CIFAR10(root='./dataset', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    # Define neural network
    net = SimpleCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    print('Start training...')
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % args.print_steps == 0:
                print(f'[{epoch+1:3d}, {i:5d}] loss: {running_loss/args.print_steps:.3f}')
                running_loss = 0.0

    torch.save(net.state_dict(), args.model_filename)
    print(f'Saved model to {args.model_filename}')

    # Test with testset
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on testset: {100 * correct // total} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-n', '--num_workers', type=int, default=4)
    parser.add_argument('-e', '--num_epochs', type=int, default=10)
    parser.add_argument('--print_steps', type=int, default=200)
    parser.add_argument('--data_path',default='./dataset/')
    parser.add_argument('--model_filename', default='cifar10_model.pth')
    args = parser.parse_args()

    main(args)