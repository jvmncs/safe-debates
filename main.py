import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import ToTensor

import argparse

from judge import Judge
from agent import Agent
from debating import Debate

def main(args):
    """main man"""
    # reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed) # don't think this works with SparseMNIST right now
        np.random.seed(args.seed)

    # cuda
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # data
    dataset = MNIST('./data/', train=False, transform=ToTensor())
    kwargs = {'num_workers': 1}
    if args.use_cuda:
        kwargs['pin_memory'] = True
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True, **kwargs)

    # load judge
    judge_state = torch.load(args.checkpoint)['state_dict']

    # debate game
    debate = Debate(data_loader, args)
    judge = Judge().to(device).load_state_dict(judge_state)
    helper = Agent(honest=True, args)
    liar = Agent(honest=False, args)




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MNIST Debate Game')
    parser.add_argument('--pixels', type=int, required=True, metavar='C',
                        help='number of nonzero pixels to uncover in sparse image (required)')
    parser.add_argument('--checkpoint', type=str, required=True, metavar='CHAR',
                        help='model checkpoint to use as judge (required)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='simlutaneous games to play (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--data-folder', type=str, default='./data/', metavar='PATH',
                        help='root path for folder containing MNIST data download \
                        (default: ./data/)')

    main(args)