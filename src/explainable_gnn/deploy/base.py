import explainable_gnn as eg
from .general import _general_deploy_method, _general_deploy2multi_device, \
    _general_add_device_to_multi_device, _general_remove_device_from_multi_device
from .device import DeployDevice, MultiAbstractDeployDevice
from .framework import DeployFramework
from .requirement import DeployRequirement


class DeployModel(eg.Model):
    def __init__(self, model: eg.Module, **kwargs):

        super(DeployModel, self).__init__()
        self.inference_method = None
        self.deploy_method = None
        self.deploy_meta = {}
        if getattr(model, "deploy_method", None) is not None:
            model.deploy_method(self, **kwargs)
        else:
            _general_deploy_method(self, **kwargs)

    def inference(self, *args, **kwargs):
        if not self.inference_method:
            raise NotImplementedError
        return self.inference_method(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.inference(*args, **kwargs)

    def deploy_device(self):
        return self.deploy_meta.get("device", None)

    def deploy_device_to(self, device):
        self.deploy_meta["device"] = device
        if getattr(self, "move_device", None) is not None:
            self.move_device(device)

    def deploy_framework(self):
        return self.deploy_meta.get("framework", None)

    def deploy_framework_to(self, framework: DeployFramework):
        self.deploy_meta["framework"] = framework
        if getattr(self, "move_framework", None) is not None:
            self.move_framework(framework)

    def deploy_requirement(self):
        """
        Get the special requirement of the deploy model
        such as the version of some packages, the memory requirement, etc.
        """
        return self.deploy_meta.get("requirement", None)

    def deploy_requirement_to(self, requirement: DeployRequirement):
        self.deploy_meta["requirement"] = requirement
        if getattr(self, "move_requirement", None) is not None:
            self.move_requirement(requirement)

    def multi_device_status(self):
        return self.deploy_meta.get("multi_device", False)

    def deploy_multi_device(self, devices: list):
        self.deploy_meta["multi_device"] = devices
        self.deploy_meta["device"] = MultiAbstractDeployDevice(devices)
        if getattr(self, "move_multi_device", None) is not None:
            self.move_multi_device(devices)
        else:
            _general_deploy2multi_device(self, devices)

    def add_deploy_device(self, device: DeployDevice):
        if self.deploy_meta.get("device", None) is None:
            self.deploy_meta["device"] = device
            if getattr(self, "move_device", None) is not None:
                self.move_device(device)
        elif self.multi_device_status() is False:
            self.deploy_multi_device([self.deploy_meta["device"], device])
        else:
            if device in self.deploy_meta["multi_device"]:
                raise ValueError("The device is already in the multi_device")
            self.deploy_meta["device"].check_vaild_device_type(device)
            self.deploy_meta["multi_device"].append(device)
            if getattr(self, "add_device_to_multi_device", None) is not None:
                self.add_device_to_multi_device(device)
            else:
                _general_add_device_to_multi_device(self, device)

    def remove_deploy_device(self, device: DeployDevice = None):
        if self.deploy_meta.get("device", None) is None:
            Warning("No device to remove")
        elif self.multi_device_status() is False:
            if self.deploy_meta["device"] != device and device is not None:
                raise ValueError("The device to remove is not the current device")
            else:
                self.deploy_meta["device"] = None
                if getattr(self, "remove_device", None) is not None:
                    self.remove_device(device)
        else:
            if device is None:
                device = self.deploy_meta["multi_device"][-1]
            if device not in self.deploy_meta["multi_device"]:
                raise ValueError("The device is not in the multi_device")
            self.deploy_meta["multi_device"].remove(device)
            if getattr(self, "remove_device_from_multi_device", None) is not None:
                self.remove_device_from_multi_device(device)
            else:
                _general_remove_device_from_multi_device(self, device)
