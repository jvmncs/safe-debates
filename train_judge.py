from __future__ import print_function
import os
from datetime import datetime
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


from prepare_data import prepare_data
from judge import Judge
from utilities import save_checkpoint, mkdir_p


def main(args):
    # reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed) # don't think this works with SparseMNIST right now
        np.random.seed(args.seed)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if args.checkpoint_filename is None:
        checkpoint_file = args.checkpoint + str(datetime.now())[:-10]
    else:
        checkpoint_file = args.checkpoint + args.checkpoint_filename

    # cuda
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # eval?
    args.evaluate = args.val_batches > 0

    # prep sparse mnist
    if not args.evaluate:
        train_loader, _, test_loader = prepare_data(args)
    else:
        train_loader, val_loader, test_loader = prepare_data(args)

    # machinery
    model = Judge().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # setup validation metrics we want to track for tracking best model over training run
    best_val_loss = float('inf')
    best_val_acc = 0

    print('\n================== TRAINING ==================')
    model.train() # set model to training mode

    # set up training metrics we want to track
    correct = 0
    train_num = args.batches * args.batch_size

    # timer
    time0 = time.time()

    for ix, (sparse, img, label) in enumerate(train_loader): # iterate over training batches
        sparse, label = sparse.to(device), label.to(device) # get data, send to gpu if needed
        optimizer.zero_grad() # clear parameter gradients from previous training update
        logits = model(sparse) # forward pass
        loss = F.cross_entropy(logits, label) # calculate network loss
        loss.backward() # backward pass
        optimizer.step() # take an optimization step to update model's parameters

        pred = logits.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(label.view_as(pred)).sum().item() # add to running total of hits

        if ix % args.log_interval == 0: # maybe log current metrics to terminal
            print('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t\
                Accuracy: {:.2f}%\tTime: {:0f} min, {:.2f} s'.format(
                (ix + 1) * len(sparse), train_num,
                100. * ix / len(train_loader),
                loss.item(),
                100. * correct / ((ix + 1) * len(sparse)),
                (time.time() - time0) // 60,
                (time.time() - time0) % 60))

    print('Train Accuracy: {}/{} ({:.2f}%)\tTrain Time: {:0f} minutes, {:2f} seconds\n'.format(
        correct, train_num, 100. * correct / train_num,
        (time.time() - time0) // 60, (time.time() - time0) % 60))

    if args.evaluate:
        print('\n================== VALIDATION ==================')
        model.eval()

        # set up validation metrics we want to track
        val_loss = 0.
        val_correct = 0
        val_num = args.eval_batch_size * args.val_batches

        # disable autograd here (replaces volatile flag from v0.3.1 and earlier)
        with torch.no_grad():
            for sparse, img, label in val_loader:
                sparse, label = sparse.to(device), label.to(device)
                logits = model(sparse)

                val_loss += F.cross_entropy(logits, label, size_average=False).item()

                pred = logits.max(1, keepdim=True)[1]
                val_correct += pred.eq(label.view_as(pred)).sum().item()

        # update current evaluation metrics
        val_loss /= val_num
        val_acc = 100. * val_correct / val_num
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_loss, val_correct, val_num, val_acc))

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss # note this is val_loss of best model w.r.t. accuracy,
                                     # not the best val_loss throughout training

        # create checkpoint dictionary and save it;
        # if is_best, copy the file over to the file containing best model for this run
        state = {
            'state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        save_checkpoint(state, is_best, checkpoint_file)

    print('\n================== TESTING ==================')
    check = torch.load(checkpoint_file + '-best.pth.tar')
    model.load_state_dict(check['state_dict'])
    model.eval()

    test_loss = 0.
    test_correct = 0
    test_num = args.eval_batch_size * args.test_batches

    # disable autograd here (replaces volatile flag from v0.3.1 and earlier)
    with torch.no_grad():
        for sparse, img, label in test_loader:
            sparse, label = sparse.to(device), label.to(device)
            logits = model(sparse)
            test_loss += F.cross_entropy(logits, label, size_average=False).item()
            pred = logits.max(1, keepdim=True)[1] # get the index of the max logit
            test_correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= test_num
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_correct, test_num,
        100. * test_correct / test_num))

    print('Final model stored at "{}".'.format(checkpoint_file + '-best.pth.tar'))


if __name__ == '__main__':
    # parses arguments when running from terminal/command line
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example Training')
    # Training settings/hyperparams
    parser.add_argument('--pixels', type=int, required=True, metavar='C',
                        help='number of nonzero pixels to remain in sparse image')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--eval-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for evaluation (default: 1,000)')
    parser.add_argument('--batches', type=int, default=30000, metavar='N',
                        help='number of batches to train on (default: 30,000')
    parser.add_argument('--val-batches', type=int, default=4, metavar='N',
                        help='batches to use for validation (default: 4)')
    parser.add_argument('--test-batches', type=int, default=800, metavar='N',
                        help='percent of non-test data to use for training (default: 800)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='log training status every N batches (default: 1000)')
    parser.add_argument('--data-folder', type=str, default='./data/', metavar='PATH',
                        help='root path for folder containing MNIST data download \
                        (default: ./data/)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/', metavar='PATH',
                        help='root path for folder containing model checkpoints \
                        (default: ./checkpoint/)')
    parser.add_argument('--checkpoint-filename', type=str, default=None, metavar='PATH',
                        help='filename for model checkpoint (default: current datetime)')
    args = parser.parse_args()

    main(args)
