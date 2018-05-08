from __future__ import print_function
import os
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


from prepare_data import prepare_data
from model import Judge
from utilities import save_checkpoint, mkdir_p


def main(args):
    # reproducibility
    # need to seed numpy/torch random number generators
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    # need directory with checkpoint files to recover previously trained models
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    checkpoint_file = args.checkpoint + args.model + str(datetime.now())[:-10]

    # decide which device to use; assumes at most one GPU is available
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # decide if we're using a validation set;
    # if not, don't evaluate at end of epochs
    evaluate = args.train_split < 1.

    # prep data loaders
    if args.train_split == 1:
        train_loader, _, test_loader = prepare_data(args)
    else:
        train_loader, val_loader, test_loader = prepare_data(args)

    # build model
    if args.model == 'linear':
        model = Softmax().to(device)
    elif args.model == 'neuralnet':
        model = TwoLayer().to(device)
    else:
        model = ConvNet().to(device)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    # setup validation metrics we want to track for tracking best model over training run
    best_val_loss = float('inf')
    best_val_acc = 0

    # loop over epochs
    for epoch in range(args.epochs):
        print('\n================== TRAINING ==================')
        model.train() # set model to training mode

        # set up training metrics we want to track
        correct = 0
        train_num = int(len(train_loader.dataset) * (1 - args.test_split) * args.train_split)

        for ix, (img, label) in enumerate(train_loader): # iterate over training batches
            img, label = img.to(device), label.to(device) # get data, send to gpu if needed
            optimizer.zero_grad() # clear parameter gradients from previous training update
            output = model(img) # forward pass
            loss = F.cross_entropy(output, label) # calculate network loss
            loss.backward() # backward pass
            optimizer.step() # take an optimization step to update model's parameters

            pred = output.max(1, keepdim=True)[1] # get the index of the max logit
            correct += pred.eq(label.view_as(pred)).sum().item() # add to running total of hits

            if ix % args.log_interval == 0: # maybe log current metrics to terminal
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (ix + 1) * len(img), train_num,
                    100. * ix / len(train_loader), loss.item()))

        # print whole epoch's training accuracy; useful for monitoring overfitting
        print('Train Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, train_num, 100. * correct / train_num))

        if evaluate:
            print('\n================== VALIDATION ==================')
            model.eval() # set model to evaluate mode

            # set up validation metrics we want to track
            val_loss = 0.
            val_correct = 0
            val_num = int(len(val_loader.dataset) * (1 - args.test_split) * (1 - args.train_split))

            # disable autograd here (replaces volatile flag from v0.3.1 and earlier)
            with torch.no_grad():
                # loop over validation batches
                for img, label in val_loader:
                    img, label = img.to(device), label.to(device) # get data, send to gpu if needed
                    output = model(img) # forward pass

                    # sum up batch loss
                    val_loss += F.cross_entropy(output, label, size_average=False).item()

                    # monitor for accuracy
                    pred = output.max(1, keepdim=True)[1] # get the index of the max logit
                    val_correct += pred.eq(label.view_as(pred)).sum().item() # add to total hits

            # update current evaluation metrics
            val_loss /= val_num
            val_acc = 100. * val_correct / val_num
            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                val_loss, val_correct, val_num, val_acc))

            # check if best model according to accuracy;
            # if so, replace best metrics
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_val_loss = val_loss # note this is val_loss of best model w.r.t. accuracy,
                                         # not the best val_loss throughout training

            # create checkpoint dictionary and save it;
            # if is_best, copy the file over to the file containing best model for this run
            state = {
                'epoch': epoch,
                'model': args.model,
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }
            save_checkpoint(state, is_best, checkpoint_file)

    print('\n================== TESTING ==================')
    # load best model from training run (according to validation accuracy)
    check = torch.load(checkpoint_file + '-best.pth.tar')
    model.load_state_dict(check['state_dict'])
    model.eval() # set model to evaluate mode

    # set up evaluation metrics we want to track
    test_loss = 0.
    test_correct = 0
    test_num = int(len(test_loader.dataset) * args.test_split)

    # disable autograd here (replaces volatile flag from v0.3.1 and earlier)
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            # sum up batch loss
            test_loss += F.cross_entropy(output, label, size_average=False).item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max logit
            test_correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= test_num
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, test_num,
        100. * test_correct / test_num))

    print('Final model stored at "{}".'.format(checkpoint_file + '-best.pth.tar'))


if __name__ == '__main__':
    # parses arguments when running from terminal/command line
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example Training')
    # Training settings/hyperparams
    parser.add_argument('--model', type=str, choices=['linear', 'neuralnet', 'convnet'],
                        required=True, metavar='CHAR',
                        help='what kind of model to train (required)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for evaluation (default: 1000)')
    parser.add_argument('--test-split', type=float, default=.2, metavar='P',
                        help='percent of training data to hold out for test set (default: .2)')
    parser.add_argument('--train-split', type=float, default=.8, metavar='P',
                        help='percent of non-test data to use for training (default: .8)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train \
                        (default: 10; ignored if model is `linear`)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001; ignored if model is `linear`)')
    parser.add_argument('--no-amsgrad', action='store_true', default=False,
                        help='don\'t use amsgrad in Adam; needed for later work with safety \
                        debates paper')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status \
                        (default: 200)')
    parser.add_argument('--data-folder', type=str, default='./data/', metavar='PATH',
                        help='root path for folder containing MNIST data download \
                        (default: ./data/)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/', metavar='PATH',
                        help='root path for folder containing model checkpoints \
                        (default: ./checkpoint/)')
    args = parser.parse_args()

    args.amsgrad = not args.no_amsgrad
    main(args)
