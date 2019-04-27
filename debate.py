import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

import agent
from utilities import AverageMeter

class Debate(object):
    def __init__(self, agents: tuple, data_loader: DataLoader, args: ArgumentParser):
        self.data_loader = data_loader
        self.pixels = args.pixels
        self.batch_size = args.batch_size
        for agent in agents:
            if agent.honest:
                self.helper = agent
            else:
                self.liar = agent
        self.first = self.liar if args.liar_first else self.helper
        self.second = self.helper if args.liar_first else self.liar
        self.reset_()
        # TODO: precommit
        # self.precommit = args.precommit
        # if self.precommit:
        #     self.commit_delta = 1
        # else:
        #     self.commit_delta = None
        
    def play(self, judge, device):
        self.images, self.labels = next(self.iterator)
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)
        sparse = torch.zeros_like(self.images).expand(-1, 2, -1, -1)

        for i in range(self.pixels):
            if i % 2 == 0:
                action = self.first.simulate_(judge, sparse, self.images, self.labels)
                self.first.reset_()
            else:
                action = self.second.simulate_(judge, sparse, self.images, self.labels)
                self.second.reset_()
            sparse = agent.Agent.apply_action(action, sparse, self.images)
        return dict(helper=self.evaluate(self.helper, judge, sparse, self.labels),
                    liar=self.evaluate(self.liar, judge, sparse, self.labels),
                    labels=self.labels)

    @classmethod
    def evaluate(cls, agent, judge, final_states, labels):
        # TODO: Handle precommit
        logits = judge(final_states)
        preds = torch.max(logits, dim=-1)[1]
        if agent.commits is None and not agent.honest:
            return dict(preds=preds, wins=(preds != labels).type(torch.float))
        elif agent.commits is not None and not agent.honest:
            raise NotImplementedError
        elif agent.commits is None and agent.honest:
            return dict(preds=preds, wins=(preds == labels).type(torch.float))
        elif agent.commits is not None and agent.honest:
            raise NotImplementedError

    def reset_(self):
        self.iterator = iter(self.data_loader)
        self.first.reset_()
        self.second.reset_()
