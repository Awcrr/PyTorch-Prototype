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

class MNISTSet:
    def __init__(self, args, split): 
        self.data_dir = args.data_dir 
        self.train = split == 'train'
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    def getset(self):
        return datasets.MNIST(self.data_dir,
            train=self.train,
            transform=self.transforms,
            download=True)
