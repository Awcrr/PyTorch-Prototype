import os

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import torch.utils.data as data

def create_loader(args, split):
    dataset = globals()[args.dataset + 'Set'](args, split)
    if args.sampler:
        sampler = globals()[args.sampler](args)
    else:
        sampler = None
    return DataLoader(
            dataset.getset(),
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True)

class MNISTSet(object):
    def __init__(self, args, split): 
        self.data_dir = args.data_dir 
        self.train = split == 'train'
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            ])

    def getset(self):
        return datasets.MNIST(self.data_dir,
            train=self.train,
            transform=self.transforms,
            download=True)

class CIFAR100Set(object):
    def __init__(self, args, split):
        self.data_dir = args.data_dir
        self.train = split == 'train'
        normalize = transforms.Normalize(mean=[x / 255. for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255. for x in [63.0, 62.1, 66.7]])
        if self.train:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize
                ])

    def getset(self):
        return datasets.CIFAR100(self.data_dir,
            train=self.train,
            transform=self.transforms,
            download=True)

class ImageNetSet(object):
    def __init__(self, args, split):
        self.data_dir = os.path.join(args.data_dir, split)
        self.train = split == 'train'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) 

        if self.train:
            self.transforms = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
                ])

    def getset(self):
        return datasets.ImageFolder(root=self.data_dir,
                transform=self.transforms) 
