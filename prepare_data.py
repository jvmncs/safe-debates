import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

class SparseMNIST(datasets.MNIST):
    def __init__(self, pixels, n, sparse_transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixels = pixels
        self._n = n
        self.sparse_transform = sparse_transform
        
    def __getitem__(self, ix):
        if self.train:
            img, target = self.train_data[ix], self.train_labels[ix]
        else:
            img, target = self.test_data[ix], self.test_labels[ix]
        img = img.unsqueeze(0).type(torch.float32) / 255
        sparse = img.clone()
        mask = torch.zeros_like(img, dtype=torch.uint8)
        n = img.numel()
        width = img.size()[1]
        pixels = torch.multinomial(torch.tensor([1]*n, dtype=torch.float), self.pixels)
        for pixel in pixels:
            pixel_x = pixel.item() % width
            pixel_y = pixel.item() // width
            mask[0][pixel_y][pixel_x] = 1
        sparse[~mask] = 0

        if self.transform is not None:
            img = self.transform(img)
            sparse = self.transform(sparse)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sparse, img, target

    def __len__(self):
        return self._n

def prepare_data(args):
    """Prepare SparseMNIST DataLoaders for training/test and optional validation"""

    # Datasets
    transform = transforms.Normalize((0.1307,), (0.3081,))
    train_set = SparseMNIST(args.pixels, args.batches,
        args.data_folder, train=True, transform=transform)
    if args.val_batches != 0:
        val_set = SparseMNIST(args.pixels, args.val_batches,
            args.data_folder, train=True, transform=transform)
    test_set = SparseMNIST(args.pixels, args.test_batches,
        args.data_folder, train=False, transform=transform)

    # DataLoaders
    kwargs = {'num_workers': 1}
    if args.use_cuda:
        kwargs['pin_memory'] = True
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
        sampler=train_sampler, **kwargs)
    if args.val_batches != 0:
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size,
            sampler=val_sampler, **kwargs)
    else:
        val_loader = None
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, **kwargs)

    return train_loader, val_loader, test_loader
