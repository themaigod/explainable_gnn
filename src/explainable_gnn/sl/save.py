import explainable_gnn as eg
from typing import Union
import torch


def save(model: Union[eg.Module, eg.InferenceModel, eg.DeployModel, eg.Translator],
         path=None,
         cloud: bool = False,
         build_info: bool = True,
         cloud_path: str = None,
         auto_inference: bool = False,
         auto_deploy: bool = False,
         **kwargs
         ) -> None:
    """
save Function
=============

The ``save`` function facilitates the saving of different types of models in the Explainable GNN framework, supporting various formats and storage locations, including local and cloud-based options.

Function Definition
-------------------
.. function:: save(model: Union[eg.Module, eg.InferenceModel, eg.DeployModel, eg.Translator], path=None, cloud: bool = False, build_info: bool = True, cloud_path: str = None, auto_inference: bool = False, auto_deploy: bool = False, **kwargs) -> None

   Saves a model to a specified path, optionally to the cloud, and can include additional build information and automation features.

   :param model: The model to save, which can be an instance of any model class within the framework.
   :param path: The local file path where the model should be saved.
   :param cloud: A boolean flag indicating whether the model should also be saved to the cloud.
   :param build_info: A boolean indicating whether to save build information alongside the model.
   :param cloud_path: The cloud storage path where the model should be saved if ``cloud`` is True.
   :param auto_inference: Automatically configures the model for inference after saving.
   :param auto_deploy: Automatically prepares the model for deployment after saving.
   :param kwargs: Additional keyword arguments that might be required by specific model saving functions.

   This function determines how to save the model based on its type and the provided parameters. If the model has a custom saving method (indicated by ``save_regular`` attribute), it uses that method; otherwise, it defaults to the appropriate saving mechanism for each model type.

Examples
--------
Saving a model locally and to the cloud:

.. code-block:: python

    model = eg.Module(some_configuration)
    save(model, path="path/to/local", cloud=True, cloud_path="path/to/cloud")

Handling different model types:

.. code-block:: python

    inference_model = eg.InferenceModel(some_other_model)
    save(inference_model, path="path/to/save", auto_inference=True)

Notes
-----
- The ``save`` function is versatile and can handle various scenarios and model types, making it essential for flexible model management within different environments.

    """
    if not model.save_regular:
        model.save(path=path, cloud=cloud, build_info=build_info, cloud_path=cloud_path,
                   auto_inference=auto_inference,
                   auto_deploy=auto_deploy, **kwargs)
    else:
        if cloud:
            eg.cloud.save(model, build_info=build_info,
                          cloud_path=cloud_path,
                          auto_inference=auto_inference,
                          auto_deploy=auto_deploy, **kwargs)
            eg.save(model, path=path, build_info=build_info, cloud=False,
                    auto_inference=auto_inference,
                    auto_deploy=auto_deploy, **kwargs)
        else:
            if isinstance(model, eg.Module):
                eg.module.save(model, path=path, build_info=build_info, **kwargs)
            elif isinstance(model, eg.Translator):
                eg.translate.save(model, path=path, build_info=build_info, **kwargs)
            elif isinstance(model, eg.InferenceModel):
                eg.inference.save(model, path=path, build_info=build_info, **kwargs)
            elif isinstance(model, eg.DeployModel):
                eg.deploy.save(model, path=path, build_info=build_info, **kwargs)
            elif isinstance(model, torch.nn.Module):
                model = eg.Module(model)
                eg.save(model, path=path, build_info=build_info, cloud=False,
                        auto_inference=auto_inference,
                        auto_deploy=auto_deploy, **kwargs)
            else:
                raise ValueError(f"Model type {type(model)} not supported for save")


def decorator_save(class_save_func):
    """
    decorator_save Function
    =======================

    The ``decorator_save`` function is a decorator designed to modify the save functionality of model classes by setting the ``save_regular`` flag to False before executing the model's save method.

    Function Definition
    -------------------
    .. function:: decorator_save(class_save_func)

       Decorates a model class's save method to adjust its behavior before execution.

       :param class_save_func: The class's save method to be decorated.
       :return: The decorated function.

       This decorator ensures that any custom logic embedded in the model's regular save process is bypassed, allowing for alternative saving strategies as defined in the external save function.

    Examples
    --------
    Using the decorator to modify a model's save method:

    .. code-block:: python

        class MyModel(eg.Module):
            @decorator_save
            def save(self, *args, **kwargs):
                # Custom save logic here
                pass

        model = MyModel()
        model.save()  # The save_regular flag is set to False before this method executes.

    Notes
    -----
    - The ``decorator_save`` provides a mechanism to ensure that models can be saved consistently, regardless of custom behaviors that might otherwise interfere with standardized saving procedures.
    """
    import functools

    @functools.wraps(class_save_func)
    def wrapper(obj, *args, **kwargs):
        obj.save_regular = False
        return class_save_func(obj, *args, **kwargs)

    return wrapper
