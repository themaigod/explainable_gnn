import torch.nn as nn
from abc import ABC


class BaseModule(ABC):
    # let nn.Module is a subclass of BaseModule
    # so that all models in explainable_gnn are also nn.Module
    # and can be used in PyTorch
    pass


BaseModule.register(nn.Module)


class Model:
    """
    Base class for all models
    It is used to quickly identify the models from explainable_gnn
    """
    pass


class Module(nn.Module, Model):
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
