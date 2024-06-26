# Development of explainable_gnn framework

## Community Help

If you have any suggestions or ideas, please feel free to release an issue. I am very happy to discuss with you.
If anyone is interested in contributing to this project, please feel free to contact me by email to involve in the
development of this project. Of course, you can also directly fork this project and pull request.
This project has a long way to go, and hopes to have more people to join in.
You can jump to the [Introduction](#Introduction) and [Framework Architecture Design](#Framework-Architecture-Design)
first to get a brief understanding of the project.

## Current Version: 0.0.1

## Development Phase: Conceptual Design

## Development Plan

Conceptual Design -> Demo Implementation of the framework pipeline -> Adjustment of the framework pipeline ->
Development of the framework modules -> Approach to all existing HIN models -> Release of the framework
-> Continuous Improvement (for unfinished additional features, like cloud) -> Extension to not only HIN models

Version: (0.0.x) -> (0.1.x) -> (0.2.x) -> (0.3.x) -> (0.x.x) -> (1.0.x) -> (1.x.x) -> (2.x.x)

## Development Plan to 1.0.0

### Priority Modules

1. Translator Module
2. Model Card Module
3. Data Module
4. Replace Module

### Future Modules

1. Cloud Module
2. Save / Load Module (It seems that acting as a transformers save / load module is quite complicated)
3. Inference Model Module
4. Deployed Model Module

## Demo and Support of HIN Models Modules

### Priority HIN Models

1. HAN
2. RGCN
3. HGT
4. MAGNN
5. Will be more

## Current Installation for Development

```angular2html
pip install --editable .
```

This command will install the package in the development mode. You can directly modify the code in the package.
Notice that the package is not installed in the site-packages, but in the current directory.
Another notice is that it may be required to install a higher version of setuptools.

## Writing Tests for the Modules

The tests are written in the `eg_tests` directory. The tests are written in the `test_*.py` files.
These files' location is the same as the modules' location. The tests have been not written yet, maybe in `hypothesis`
style.

# Introduction

People developed a lot of GNN models to solve Heterogeneous Information Network (HIN) tasks. These models capture
the rich semantics of HINs. However, it is still not clear that how these models focus on constructing particular node
representations from the complex structure of HINs, which is the key to interpretability and explainability of them.
In this repository, we provide a framework translating the models to directly node connections, like one layer`GCN`,
`GAT`, etc. It is contributed to understand the node dependencies with each other and accelerate the speed of the
inference when the model is deployed. Based on the analysis of current models, we provide a simple and efficient method
to construct the node representations for regular HINs.

# Framework Architecture Design

## Overview

1. Tagging the model with meta information (by the user)
2. Replacing the model with the replace module (by the framework Translator Module)
	1. Initializing the replace module with the meta information
	2. Calculating the parameters of the replace module
	3. Replacing the model with the replace module
3. Visualizing the final replace structure (by the framework Translator Module)
4. Visualizing / Analyzing the node representation dependencies (by the framework Translator Module)
5. Approximating the model to further accelerate the inference (by the framework Translator Module)
6. Managing the model
	1. Model Type Transformation (Model -> Inference Model -> Deployed Model)
	2. Model Saving and Loading (by the framework Save / Load Module)
	3. Managing the Nodel by Model Name and Model Card (by the framework Model Info Module)
	4. Cloud Management (List, Delete, Download, Upload by the framework Cloud Module)

## Meta Information

The framework requires the user to tag the model with meta information. It is inefficient to automatically analyze the
model and generate the meta information. But the framework provides the existing common replace modules and the user
can directly use them. Besides, the user can provide the extra information to control the replacement.

`model.meta_info: dict`:

- `replace`: the replace module, type: `eg.Module`

## Data

All kinds of data can be provided separately to the translator. Besides, we also provide an unified data structure to
store the data as the only required data.

## Translator

The replace process is done by the translator. It is not only simply replacing the model but also calculating the
parameters of the replace module. The translator can visualize the final replace structure and analyze the node
representation dependencies. The translator can also approximate the model to further accelerate the inference.

`translator.replace()`: replace the model with the replace module
`translator.visualize()`: visualize the final replace structure
`translator.visualize(node_id)`: visualize the node representation dependencies
`translator.approximate()`: approximate the model to further accelerate the inference
`translator.get_model()`: get the replaced model

## Model

The translated model can be provided by the translator, inherited from the framework model module (also a
torch.nn.Module).
The model can be directly used and trained. (`eg.Module`)

## Save / Load

The model can be saved and loaded. The model can be saved with the building information card. Framework can only load
the model card.
It can be saved by `.save()` or `eg.save(model, path)`. The model can be loaded by `.autoload()` or `eg.autoload(path)`.

## Model Info

The model can be managed by the model name and the model card. The model card can be used to show the information of the
model.

Valid model name: `str` or `str:{version}`
Valid model card: `eg.ModelCard`
Model card can be used to show the information of the model: `model_card.show()`
Particular information contained in the model card:

- `model_card.model_name`: the name of the model
- `model_card.model_version`: the version of the model

## Inference Model

By calling `eg.InferenceModel(model)`, the model can be transformed to the inference model. It cannot be trained but
can be used to different situations.

## Deployed Model

Considering the different deployment requirements, the model can be deployed to different situations, by calling
`eg.deploy(model, **kwargs)`.

## Cloud

We design a cloud system to simply manage the model. It is allowed public and private cloud. The cloud can list, delete,
download, and upload the model.

`eg.cloud.*`: cloud operation
When save or load the model, the cloud can be used by setting the `cloud` and `cloud_path` parameters.

We also provide easy private cloud initialization. The user can easily set up the private cloud by
`eg.cloud.init(port, password)`.

# How to use

## Installation

[//]: # (github.com/themaigod/explainable_gnn)

```angular2html
pip install git+https:github.com/themaigod/explainable_gnn.git
```

## Example

Use the provided module to replace the model.

```python
import torch.nn as nn
import explainable_gnn as eg
import torch


class SimpleGCN(nn.Module):
    # add meta information
    meta_info = {
        'replace': eg.GCN("HAN"),
        'required': "train",
        'train_info': {
            'original_params': torch.load("path/to/params"),
            'with_input': True,
        },
        'approximate': True,
    }  # This allows the framework to replace the model with the GCN model

    def __init__(self, in_dim, out_dim):
        super(SimpleGCN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)  # only for example, not the real GCN model

    def forward(self, x, adj):
        return self.linear(x)
```

You can design your own replace module and add the meta information to the model. The framework will automatically
replace the model with the provided module.

```python
import torch.nn as nn
import explainable_gnn as eg


class ReplaceModule(eg.Module):
    def __init__(self, model_name):
        super(ReplaceModule, self).__init__(model_name)

    def forward(self, x, adj):
        return x

    def parameters_calculation(self, **kwargs):
        return  # return the parameters of the model
```

After adding the meta information, now you can use the framework to replace the model with the provided module.
Here are some examples of how to use the framework.

```python
import torch.nn as nn
import explainable_gnn as eg


class OriginalModel(nn.Module):
    pass


# translator operation
original_model = OriginalModel()
translator = eg.Translator(original_model)
translator.replace()
translator.visualize()
translator.visualize(node_id=0)
translator.approximate()  # if required
improved_model = translator.get_model()

# model operation
output = improved_model(input)  # directly use, still can be trained
improved_model.save("path/to/model", build_info=True)  # save the model with building a information card
inferenced_model = eg.InferenceModel(improved_model, backend="cpu")
# get inference model that can be deployed in different situations
inferenced_model(input)  # directly use, cannot be trained
# if have specific deployment requirements
inferenced_model = eg.deploy(improved_model, **kwargs)
inferenced_model.save("path/to/model", build_info=True)
# load the model
inferenced_model = eg.autoload("path/to/model", inferenced=True)
build_info = eg.autoload("path/to/model", only_info=True)
build_info.show()
build_info = inferenced_model.build_info()

# cloud save
improved_model.save("name", cloud=True, cloud_path="path/to/cloud", build_info=True,
                    auto_inference=True, auto_deploy=True, **kwargs)

improved_model = eg.autoload("name", cloud=True, cloud_path="path/to/cloud")
inferenced_model = eg.autoload("name", cloud=True, cloud_path="path/to/cloud", inferenced=True)
deployed_model = eg.autoload("name", cloud=True, cloud_path="path/to/cloud", deploy=True, **kwargs)
build_info = eg.autoload("name", cloud=True, cloud_path="path/to/cloud", only_info=True)

# cloud operation
eg.cloud.list(cloud_path="path/to/cloud")
eg.cloud.delete("name", cloud_path="path/to/cloud")
eg.cloud.download("name", cloud_path="path/to/cloud", local_path="path/to/local")
eg.cloud.upload("name", cloud_path="path/to/cloud", local_path="path/to/local")
```
