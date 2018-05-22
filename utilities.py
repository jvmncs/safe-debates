import os
import shutil
import torch

class BatchDict(dict):
    """A dict-like that takes care of batching with key-dependent defaults"""
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get = super(self.__class__, self).__getitem__
        self._set = super(self.__class__, self).__setitem__
        self.device = device

    def __getitem__(self, keys):
        values = [self._get(self._to_tuple(key)).unsqueeze(0) for key in keys]
        return torch.cat(values, dim=0)

    def __setitem__(self, keys, values):
        for key, value in zip(keys, values):
            self._set(self._to_tuple(key), value)

    def __missing__(self, key):
        """
        Default to 1d FloatTensor of zeros with similar length.
        """
        return torch.zeros(len(key), dtype=torch.float, device=self.device)

    @classmethod
    def _to_tuple(cls, key):
        """Convert n-d tensor to flattened tuple for key storage"""
        return tuple(key.contiguous().view(-1).tolist())

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def track_stats_(total_meter, class_meters, preds, wins, labels, precommit):
    """Track stats"""
    if precommit:
        track_stats_with(total_meter, class_meters, preds, wins, labels)
    else:
        track_stats_without(total_meter, class_meters, preds, wins, labels)

def track_stats_without(total_meter, class_meters, preds, wins, labels):
    """Track stats without precommit"""
    correct = wins.type(torch.float).sum().item()
    total_meter.update(correct)
    unique = set([i for i in labels.tolist()])
    for i in sorted(unique):
        for j in ((preds == i) + (labels == i)).eq(2):
            val = float(j.item())
            total_meter.update(val)
            class_meters[i].update(val)

def track_stats_with(total_meter, class_meters, wins, labels):
    """Track stats with precommit"""
    raise NotImplementedError

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_checkpoint(state, is_best, title, filename='checkpoint.pth.tar'):
    filepath = title + '-' + filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, title + '-best.pth.tar')
