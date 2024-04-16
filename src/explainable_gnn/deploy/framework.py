from typing import List, Union


class DeployFramework:
    framework_name = None

    def __init__(self, **kwargs):
        self.framework_meta = {}
        if kwargs:
            self.framework_meta.update(kwargs)
        if not self.framework_name:
            self.framework_name = "default_framework"


class PyTorchDeployFramework(DeployFramework):
    framework_name = "pytorch"


class NumPyDeployFramework(DeployFramework):
    framework_name = "numpy"


class MultiAbstractDeployFramework(DeployFramework):
    framework_name = "multi_framework"

    def __init__(self, frameworks: List[DeployFramework], **kwargs):
        self.frameworks = frameworks
        super().__init__(**kwargs)
        self.check_vaild_framework_type(frameworks)

    def check_vaild_framework_type(self, framework: Union[
        DeployFramework, List[DeployFramework]]
                                   ):
        if isinstance(framework, list):
            for f in framework:
                if self.framework_meta.get("strict", False):
                    try:
                        assert f.framework_name in self.frameworks, \
                            "The framework is not in the multi_framework"
                    except AttributeError:
                        raise ValueError(
                            "The framework is not a registered DeployFramework")
                else:
                    if f.framework_name not in self.frameworks:
                        register_deploy_framework(f)
        else:
            if self.framework_meta.get("strict", False):
                try:
                    assert framework.framework_name in self.frameworks, \
                        "The framework is not in the multi_framework"
                except AttributeError:
                    raise ValueError(
                        "The framework is not a registered DeployFramework")
            else:
                if framework.framework_name not in self.frameworks:
                    register_deploy_framework(framework)

    def exist_framework(self, framework):
        return framework in self.frameworks


def register_deploy_framework(framework: DeployFramework):
    registered_deploy_framework[framework.framework_name] = framework
    return framework


registered_deploy_framework = {
    "pytorch": PyTorchDeployFramework,
    "numpy": NumPyDeployFramework
}
