import explainable_gnn as eg
from typing import Any, Dict, List, Tuple, Union
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
