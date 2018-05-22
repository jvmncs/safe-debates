from __future__ import print_function
import gc
import torch
from argparse import ArgumentParser
import tqdm

from debate import Debate
from utilities import BatchDict

class Agent(object):
    def __init__(self, honest: bool, args: ArgumentParser):
        self.honest = honest
        self.pixels = args.pixels
        self.rollouts = args.rollouts
        self.P = 1. / args.pixels
        self.N = BatchDict(args.device) # visit count
        self.Q = BatchDict(args.device) # mean state-action value

    def rollout_(self, judge, states, images, labels, level):
        if level > self.pixels:
            return -Debate.evaluate(self, judge, states, labels)['wins']

        valids = self.get_valid_actions(states, images).type(torch.float)
        Us = self.break_ties(self.calculate_UCT(states))
        obj = self.Q[states] + Us
        actions = torch.max(obj * valids, dim=-1)[1]
        successors = self.apply_action(actions, states, images)
        Vs = self.rollout_(judge, successors, images, labels, level=level + 1)

        # TODO: vectorize this
        for action in actions:
            self.Q[states][:, action] = ((self.N[states][:, action] * self.Q[states][:, action] + Vs) /
                (self.N[states][:, action] + 1))
            self.N[states][:, action] += 1

        return -Vs

    def simulate_(self, judge, roots, images, labels):
        """
        Performs a certain number of rollouts from the root of a vectorized set of debate trees.
        """
        for i in tqdm.tqdm(range(self.rollouts)):
            self.rollout_(judge, roots, images, labels, level=0)
        return self.choose_moves(roots, images)

    def choose_moves(self, states, images):
        valids = self.get_valid_actions(states, images).type(torch.float)
        available_action_counts = self.N[states] * valids
        return torch.max(available_action_counts, dim=-1)[1]

    def get_valid_actions(self, states, images):
        unrevealed = states == 0
        nonzero = images > 0
        return (unrevealed + nonzero == 2).view(states.size(0), -1)

    def calculate_UCT(self, states):
        P = 1. / (
            states.contiguous().view(states.size(0), -1) > 0).sum(-1).sum(-1).type(torch.float)
        Ns = self.N[states]
        return (P * Ns.sum(-1).sqrt()).unsqueeze(1) / (1 + Ns)

    @classmethod
    def break_ties(cls, UCT):
        mxs, ixs = torch.max(UCT, dim=-1, keepdim=True)
        needs_random = (UCT == mxs).sum(-1, keepdim=True) > 1
        ixs[needs_random] = 0
        random_ints = torch.randint_like(ixs, UCT.size(1))
        ixs += needs_random.type(torch.long) * random_ints
        return ixs.type(torch.float)

    @classmethod
    def apply_action(cls, action, sparse, images):
        width = images.size(-1)
        action_x = action % width
        action_y = action / width
        sparse[:, 0, action_x, action_y] = 1
        sparse[:, 1, action_x, action_y] = images[:, 0, action_x, action_y]
        return sparse

    def precommit_(self, commits, opponent_commits):
        self.commits = commits
        self.opponent_commits = opponent_commits

    def reset_(self):
        self.N.clear()
        self.Q.clear()
        gc.collect();
