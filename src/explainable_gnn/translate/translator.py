from explainable_gnn import BaseModule
import explainable_gnn as eg
import torch


class Translator(eg.Module):
    def __init__(self, model: BaseModule, data=None, hetero=None):
        super().__init__()
        self.model = model
        self.model_use = "model"

        self.hetero = hetero
        self.standard_data = None
        if data is not None:
            if not isinstance(data, eg.Data):
                raise ValueError("data should be a Data object")
            self.data = data
            if isinstance(data, eg.HINData):
                self.hetero = True
            elif isinstance(data, eg.DirectedGraphData):
                self.hetero = False

            if isinstance(data, eg.StandardData):
                self.standard_data = True

    def replace(self, *args, **kwargs):
        model = getattr(self, self.model_use)
        replace_model = eg.replace(model, *args, **kwargs)
        self.replace_model = replace_model
        self.model_use = "replace_model"
        self.parameters_calculation()

    def parameters_calculation(self):
        model = getattr(self, self.model_use)
        eg.parameters_calculation(model, data=self.data, hetero=self.hetero
                                  , standard_data=self.standard_data)

    def visualize(self, node_id=None, *args, **kwargs):
        model = getattr(self, self.model_use)
        if node_id is None:
            eg.structure_visualize(model, *args, **kwargs)
        else:
            eg.node_dependency(model, node_id, *args, **kwargs)

    def inference(self, *inputs):
        model = getattr(self, self.model_use)
        previous_mode = model.training
        model.eval()
        with torch.no_grad():
            output = model(*inputs)
        model.train(previous_mode)
        return output

    def get_model(self):
        return getattr(self, self.model_use)

    def approximate(self, *args, **kwargs):
        model = getattr(self, self.model_use)
        approximate_model = eg.replace(model, *args, **kwargs)
        self.approximate_model = approximate_model
        self.model_use = "approximate_model"
        self.parameters_calculation()



