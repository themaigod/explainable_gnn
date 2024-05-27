import explainable_gnn as eg


class MetaAnalysis(eg.Module):
    def __init__(self, model: eg.Module, **kwargs):
        """
        Meta Analysis for the model
        if the model has the meta_info attribute, it will be used
        otherwise, you read the use_meta_info attribute to decide

        When using the meta_info and it conflicts with inner parameters of the replace module,
        the meta_info have the priority
        Parameters
        ----------
        model
        kwargs
        """
        super(MetaAnalysis, self).__init__()
        if getattr(model, "meta_info", None) is not None:
            self.meta_info = model.meta_info
            self.use_meta_info = True
            self.model = model
        else:
            self.meta_info = {}

    def __call__(self, *args, **kwargs):
        if not self.use_meta_info:
            raise AttributeError("The model does not have the meta_info attribute")
        else:
            # TODO: implement the meta analysis
            pass
