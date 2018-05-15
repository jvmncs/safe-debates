class Debate(object):
    def __init__(self, data_loader, args):
        self.data_loader = data_loader
        self.batch_size = args.batch_size

        self.reset(agents)

    def reset(self, agents):
        self.iterator = iter(self.data_loader)
        self.sparse, self.image, self.label = next(self.iterator)
        for agent in agents:
            if agent.honest:
                self.helper = agent
            else:
                self.liar = agent

    @classmethod
    def choose_winner(cls, judge, final_image):
        logits = judge(final_image)
        probs = nn.functional.softmax(logits, dim=-1)
        torch.max(probs, dim=-1)
