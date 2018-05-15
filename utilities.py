import os
import shutil
import torch

class BatchDict(dict):
    """A dict-like that takes care of batching with key-dependent defaults"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get = super(self.__class__, self).__getitem__
        self._set = super(self.__class__, self).__setitem__

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
        return torch.zeros(len(key), dtype=torch.float)

    @classmethod
    def _to_tuple(cls, key):
        """Convert n-d tensor to flattened tuple for key storage"""
        return tuple(key.view(-1).tolist())


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
