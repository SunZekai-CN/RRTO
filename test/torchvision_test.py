import argparse
import os.path as osp
import os
import shutil
import time
import warnings
import sys
import random

import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
from torchvision import transforms, datasets, models
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import PILToTensor
from inspect import getmembers, isclass
project_dir = osp.abspath(osp.join(*([osp.abspath(__file__)] + [os.pardir])))


tasks = ["all", "classification", "detection", "segmentation", "video"]
parser = argparse.ArgumentParser(description='torch vision inference')
parser.add_argument('-t', '--task', default='classification',
                    choices=tasks,
                    help='task: ' +
                    ' | '.join(tasks) +
                    ' (default: classification)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                    help='model architecture')
parser.add_argument('-d', '--dataset', default='CIFAR10',
                    help='dataset')
parser.add_argument('-p', '--parallel', default='all',
                    help='parallel approach')
parser.add_argument('-ip', '--ip', default='127.0.0.1',
                    help='server ip')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_model_weights(model_arch: str, module):
    all_class = getmembers(module, isclass)
    model_arch = model_arch.upper()
    cls = None
    for cls_name, cls in all_class:
        if cls_name.upper().startswith(model_arch):
            break
    assert cls is not None, f"{model_arch} not found."
    return cls



if __name__ == "__main__":
    args = parser.parse_args()
    task = args.task
    model_arch = args.arch
    dataset_name = args.dataset
    parallel_approach: str = args.parallel
    data_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "data", dataset_name)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using data dir: {data_dir}; parallel_approach {parallel_approach}")
    print(f"device {device}; task {task}; model_arch {model_arch}; dataset {dataset_name}")

    if False: # TODO support iterate through all these models and datasets
        if task == "all":
            all_models = models.list_models()
        if task == "classification":
            all_models = models.list_models(module=models)
        else:
            all_models = models.list_models(module=getattr(models, task))

        datasets = {
            "classification": ["CIFAR10", "CIFAR100", "MNIST"],
            "detection": ["CocoDetection", "Kitti", "VOCDetection"],
            "segmentation": ["Cityscapes", "VOCSegmentation"],
            "video": ["HMDB51", "Cityscapes"]
        }
    
    if task != "classification":
        models = getattr(models, task)
    weights = get_model_weights(model_arch, models).DEFAULT
    preprocess = weights.transforms()
    model = getattr(models, model_arch)(weights=weights)
    model.eval()
    model = model.to(device)

    if dataset_name == "ImageNet":
        kwargs = {"split": "val"}
    elif "CIFAR" in dataset_name:
        kwargs = {"download": True, "train": False}
    elif "OxfordIIITPet" in dataset_name:
        kwargs = {"download": True}

    dataset: datasets.DatasetFolder = getattr(datasets, dataset_name)(
        data_dir, transform=preprocess, **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    inference_time=AverageMeter()
    for i, (inp, target) in enumerate(dataloader):
        stime = time.time()
        pred = model(inp.to(device))
        inference_time.update(time.time() - stime)
        if i % 20 == 0:
            print(f"inference time: {inference_time.val:.3f} ms, average {inference_time.avg:.3f} ms")
        if i > 20:
            break
'''
python3 torchvision_test.py -a resnet101 -t classification
python3 torchvision_test.py -a convnext_large  -t classification
python3 torchvision_test.py -a fcn_resnet50 -t segmentation
python3 torchvision_test.py -a deeplabv3_resnet50 -t segmentation
python3 torchvision_test.py -a fasterrcnn_resnet50_fpn_v2 -t detection
python3 torchvision_test.py -a retinanet_resnet50_fpn_v2 -t detectio
'''