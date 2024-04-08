import torch.nn as nn


class Module(nn.Module):
    def parameters_calculation(self, *args, **kwargs):
        raise NotImplementedError

    def approximate(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError

    def inference(self, *args, **kwargs):
        raise NotImplementedError

    def deploy(self, *args, **kwargs):
        raise NotImplementedError
