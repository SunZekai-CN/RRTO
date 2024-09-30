# 导入必要的库
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time

BATCH_SIZE = 1
device = torch.device('cuda')
# device = torch.device("cpu")

mean = [x/255 for x in [125.3, 23.0, 113.9]]
std = [x/255 for x in [63.0, 62.1, 66.7]]

def conv3x3(in_features, out_features):
    return nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)


# VGG19
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 1
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 5
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 8
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 9
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 11
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 12
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 15
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 16
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )

        self.classifier = nn.Sequential(
            # 17
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 18
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 19
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def eval(model, dataloader):

    model.eval()
    
    accuracy = 0
    inference_time=AverageMeter()
    tag = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            print(f"input size: {batch_x.nelement()*batch_x.element_size()/1024/1024} MB")
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            start = time.time()
            logits = model(batch_x)
            end = time.time()

            probs, pred_y = logits.data.max(dim=1)
            accuracy += (pred_y==batch_y.data).float().sum()/batch_y.size(0)
            inference_time.update((end-start)*1000)
            print(f"inference time: {inference_time.val:.3f} ms, average {inference_time.avg:.3f} ms")
            if tag == 10:
                break
            tag = tag + 1

    accuracy = accuracy*100.0/len(dataloader)
    return accuracy

if __name__ == '__main__':
    print('VGG19 Using device: {}'.format(device))
    test_set = dsets.CIFAR10(root='/workspace/data/vgg19/CIFAR10/',
                             train=False,
                             download=True,
                             transform=trans.Compose([
                                trans.ToTensor(),
                                trans.Normalize(mean, std)
                            ]))

    test_dl = DataLoader(test_set,
                         batch_size=BATCH_SIZE,
                         num_workers=0) 
    vgg19 = VGG().to(device)
    eval(vgg19, test_dl)