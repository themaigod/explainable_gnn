import explainable_gnn as eg
from .general import general_inference_method


class InferenceModel(eg.Model):
    def __init__(self, model: eg.Module, **kwargs):
        self.inference_method = None
        self.deploy_method = None
        if getattr(model, "inference_method", None) is not None:
            model.inference_method(self, **kwargs)
        else:
            general_inference_method(self, **kwargs)

    def inference(self, *args, **kwargs):
        if not self.inference_method:
            raise NotImplementedError
        return self.inference_method(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.inference(*args, **kwargs)

    def deploy(self, *args, **kwargs):
        self.deploy_method(*args, **kwargs)
