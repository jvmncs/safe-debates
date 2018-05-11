import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

class SparseMNIST(datasets.MNIST):
    def __init__(self, pixels, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixels = pixels
        self._n = n
        self._generator = torch.default_generator
        
    def __getitem__(self, ix):
        if self.train:
            img, target = self.train_data[ix], self.train_labels[ix]
        else:
            img, target = self.test_data[ix], self.test_labels[ix]
            
        img = img.unsqueeze(0).type(torch.float32) / 255
        sparse = img.clone()
        mask, pixels = self._get_mask(img)
        sparse[~mask] = 0

        if self.transform is not None:
            img = self.transform(img)
            sparse = self.transform(sparse)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.cat((mask.type(torch.float32), sparse)), img, target

    def __len__(self):
        return self._n

    def _get_mask(self, img):
        mask = torch.zeros_like(img, dtype=torch.uint8)
        width = img.size()[1]
        pixels = self._get_pixels(img)
        
        for pixel in pixels:
            pixel_x = pixel.item() % width
            pixel_y = pixel.item() // width
            mask[0][pixel_y][pixel_x] = 1
        return mask, pixels

    def _get_pixels(self, img):
        nonzero = (img > 0).view(-1)
        indices = torch.arange(len(nonzero))[nonzero].type(torch.long)
        choices = torch.multinomial(torch.tensor([1] * indices.size(0), dtype=torch.float), self.pixels, generator=self._generator)
        return indices.index_select(0, choices)

def prepare_data(args):
    """Prepare SparseMNIST DataLoaders for training/test and optional validation"""
    # Datasets
    train_set = SparseMNIST(args.pixels, args.batches, args.data_folder, train=True)
    if args.evaluate:
        val_set = SparseMNIST(args.pixels, args.val_batches, args.data_folder, train=True)
    test_set = SparseMNIST(args.pixels, args.test_batches, args.data_folder, train=False)

    # DataLoaders
    kwargs = {'num_workers': 1}
    if args.use_cuda:
        kwargs['pin_memory'] = True
    train_loader = DataLoader(train_set, batch_size=args.batch_size, **kwargs)
    if args.evaluate:
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, **kwargs)
    else:
        val_loader = None
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, **kwargs)

    return train_loader, val_loader, test_loader
