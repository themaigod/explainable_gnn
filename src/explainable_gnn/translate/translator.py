from explainable_gnn.module import BaseModule
import explainable_gnn as eg


class Translator(eg.Module):
    def __init__(self, model: BaseModule, *args, **kwargs):
        super().__init__()
        self.model = model
        self.translator_method = None
