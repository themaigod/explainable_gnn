from typing import Union


class DeployDevice:
    device_name = None
    device_type = None

    def __init__(self, **kwargs):
        self.device_meta = {}
        if kwargs:
            self.device_meta.update(kwargs)
        if not self.device_name:
            self.device_name = "default_device"


class MultiAbstractDeployDevice(DeployDevice):
    device_name = "multi_device"
    device_type = "multi_device: {}"

    def __init__(self, devices: list, **kwargs):
        self.devices = devices
        super().__init__(**kwargs)
        device_type = self._get_device_type()
        self.device_type = self.device_type.format(device_type)

    def _get_device_type(self):
        device_types = [device.device_type for device in self.devices]
        # check if all the devices are the same type
        if all(device_type == device_types[0] for device_type in device_types):
            return device_types[0]
        elif self.device_meta.get("strict", False):
            raise ValueError("All devices should be the same type")
        else:
            return "mixed_device"

    def check_vaild_device_type(self, device):
        assert device.device_type == self.device_type, \
            "The device type is not the same as the multi_device type"

    def exist_device(self, device):
        return device in self.devices


class CPUDeployDevice(DeployDevice):
    device_name = "cpu"
    device_type = "cpu"


class GPUDeployDevice(DeployDevice):
    device_name = "gpu"
    device_type = "gpu"

    def __init__(self, device_id: Union[int, str], **kwargs):
        super().__init__(**kwargs)
        if isinstance(device_id, str):
            self.device_name = device_id
        else:
            self.device_name = f"gpu:{device_id}"


def register_deploy_device(device: DeployDevice):
    registered_deploy_device[device.device_name] = device
    return device


registered_deploy_device = {
    "cpu": CPUDeployDevice(),
    "gpu:0": GPUDeployDevice(0)
}
