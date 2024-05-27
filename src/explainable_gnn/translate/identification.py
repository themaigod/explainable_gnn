import explainable_gnn as eg
import torch


def _catch_replace_module(model):
    """
    Catch the replace module of the model
    Currently, it provides two ways to get the replace module:
    1. If the model is a subclass of eg.Module and has the attribute replace_module
    2. If the model is a subclass of eg.BaseModule (including torch.nn.Module) and has the meta_info attribute with the key "replace_module"
    3. otherwise, assumes the model itself is the replace module

    The priority is 1 > 2 > model itself

    """
    if isinstance(model, eg.Module) and hasattr(model, "replace_module"):
        return model.replace_module
    elif isinstance(model, eg.BaseModule) and hasattr(model,
                                                      "meta_info") and model.meta_info.get(
        "replace_module", None) is not None:
        return model.meta_info["replace_module"]
    else:
        return model


def _analyze_model_structure(model: torch.nn.Module, operation=None, depth=-1,
                             first=False
                             ):
    if first:
        skip_children, model = operation(model, None, model)
        if skip_children:
            return model

    for name, submodule in model.named_children():
        if operation is not None:
            skip_children = operation(model, name, submodule)
            if skip_children:
                continue
        if depth > 0:
            _analyze_model_structure(submodule, operation, depth - 1)

    return model
