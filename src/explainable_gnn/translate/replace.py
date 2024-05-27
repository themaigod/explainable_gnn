from .identification import _analyze_model_structure, _catch_replace_module
from .meta import MetaAnalysis
import explainable_gnn as eg


class ReplaceAnalysis(eg.Module):
    def __init__(self, model: eg.Module, **kwargs):
        """
        To analyze the needed particular operations of the replace module
        It is not allowed to use for the meta_info way, please use MetaAnalysis instead
        Parameters
        ----------
        model
        kwargs
        """
        super(ReplaceAnalysis, self).__init__()
        if getattr(model, "replace_module", None) is not None:
            self.replace_module = model.replace_module
            self.use_replace_module = True
            self.model = model
        else:
            self.replace_module = model
            self.use_replace_module = False
            self.model = model

    def __call__(self, *args, **kwargs):
        # TODO: implement the replace analysis
        pass


def replace_operation_stream(*args, **kwargs):
    def replace_operation(model, name, submodule):
        # special operation for the model itself
        if submodule is model:
            replace_module = _catch_replace_module(model)
            if replace_module is not None:
                return True, replace_module
            else:
                return False, model

        replace_module = _catch_replace_module(submodule)
        if replace_module is not None:
            setattr(model, name, replace_module)
            meta_analysis = MetaAnalysis(replace_module)
            if not meta_analysis.use_meta_info:
                replace_analysis = ReplaceAnalysis(replace_module)
                replace_analysis(*args, **kwargs)
            else:
                meta_analysis(*args, **kwargs)
            skip_children = True
        else:
            skip_children = False
        return skip_children

    return replace_operation


def replace(model, replace_stream=replace_operation_stream, *args, **kwargs):
    """
    Replace the model with the replace module
    Notice that it will not change the model itself if model has the replace module for itself
    This will return a new model with the replace module
    Otherwise, it actually returns the model itself, but with the replaced children
    Parameters
    ----------
    replace_stream: a callable which returns a callable (input: model, name, submodule)
    model
    args
    kwargs

    Returns
    -------

    """
    operation_stream = replace_stream(*args, **kwargs)
    model = _analyze_model_structure(model, operation_stream)
    return model
