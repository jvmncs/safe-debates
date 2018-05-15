from __future__ import print_function
import gc
import torch
from utilities import BatchDict

class Agent(object):
    def __init__(self, honest, target_class, args):
        self.honest = honest
        self.pixels = args.pixels
        self.P = 1. / args.pixels
        self.N = BatchDict()

    def rollout_(self, states):
        for state in torch.split(states, dim=0):
            self.N[state][action] += 1
        raise NotImplementedError

    def simulate_(self, simulations):
        for i in range(simulations):
            self.rollout_()

    def choose_moves(self, states):
        states = torch.split(states, dim=0)
        return torch.cat(torch.max(self.N[state], dim=-1)[1])
    
    def precommit_(self, target_class):
        self.target = target_class

    def reset(self):
        """Resets visit count"""
        self.N.clear()
        del self.target
        gc.collect();

    def calculate_UCT(self, states, actions):
        P = 1. / (states.view(states.size(0), -1) > 0).sum(-1).sum(-1).type(torch.float)
        Ns = self.N[states]
        print((P * Ns.sum(-1).sqrt()).size())
        print((1 + Ns).size())
        return (P * Ns.sum(-1).sqrt()).unsqueeze(1) / (1 + Ns)

    @classmethod
    def break_ties(cls, UCT):
        mxs, ixs = torch.max(UCT, dim=-1, keepdim=True)\
        needs_random = (UCT == mxs).sum(-1, keepdim=True) > 1
        ixs[needs_random] = 0
        random_ints = torch.randint_like(ixs, UCT.size(1))
        ixs += needs_random.type(torch.long) * random_ints
        return ixs
