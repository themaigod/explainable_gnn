from explainable_gnn.module import BaseModule


class Translator:
    def __init__(self, model: BaseModule):
        self.model = model
        self.translator_method = None
