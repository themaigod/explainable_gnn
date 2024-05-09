from typing import Union
import explainable_gnn as eg


class DeployDevice(eg.Model):
    """
    DeployDevice Class
    ==================

    The ``DeployDevice`` class is designed to handle the deployment settings for devices within a system. It provides a structured way to define and store device metadata dynamically.

    Attributes
    ----------
    device_name : str or None
        The name of the device. Defaults to 'default_device' if not specified during initialization.
    device_type : NoneType
        Placeholder for specifying the type of the device. Intended to be defined in subclasses or instances as needed.

    Constructor
    -----------
    .. method:: DeployDevice.__init__(**kwargs)

       Initializes a new instance of the ``DeployDevice`` class, optionally setting device metadata.

       :param kwargs: Arbitrary keyword arguments representing device metadata. These are stored in a dictionary attribute 'device_meta'.

       After initialization, if 'device_name' is not explicitly set, it defaults to "default_device".

    Attributes
    ----------
    device_meta : dict
        A dictionary to store metadata about the device. This can include specifications like model number, capabilities, configuration settings, and so on.

    Methods
    -------
    .. method:: DeployDevice.__init__(**kwargs)

       Constructs a ``DeployDevice`` instance with the provided metadata. The constructor allows for flexible specification of any number of attributes via keyword arguments.

       Example usage:

       .. code-block:: python

           device = DeployDevice(ip_address="192.168.1.1", port=8080)
           print(device.device_meta)  # Outputs: {'ip_address': '192.168.1.1', 'port': 8080}

    Notes
    -----
    - The ``DeployDevice`` class is designed to be a base class for specific types of devices. By default, it provides a flexible framework for device metadata but does not enforce any specific fields except for the device name.
    - Users can extend this class to include more specific attributes and methods tailored to particular types of devices or deployment needs.


    """
    device_name = None
    device_type = None

    def __init__(self, **kwargs):
        self.device_meta = {}
        if kwargs:
            self.device_meta.update(kwargs)
        if not self.device_name:
            self.device_name = "default_device"


class MultiAbstractDeployDevice(DeployDevice):
    """
    MultiAbstractDeployDevice Class
    ===============================

    The ``MultiAbstractDeployDevice`` class extends the ``DeployDevice`` class to manage configurations for multiple devices, allowing for operations and checks across a collection of devices.

    Attributes
    ----------
    device_name : str
        The name of the device set to "multi_device" to indicate that this instance handles multiple devices.
    device_type : str
        A format string "multi_device: {}" that will be completed based on the types of devices managed. It supports homogeneous or mixed device types.

    Constructor
    -----------
    .. method:: MultiAbstractDeployDevice.__init__(devices: list, **kwargs)

       Initializes a new instance of the ``MultiAbstractDeployDevice`` class. It sets up device management for multiple devices by inheriting and extending the ``DeployDevice``.

       :param devices: A list of device instances that this multi-device manager will handle.
       :param kwargs: Arbitrary keyword arguments that are passed to the base class and might include metadata about the collective setup of devices.

       The constructor also determines the type of devices managed by evaluating whether they are homogeneous or mixed and sets the ``device_type`` accordingly.

    Methods
    -------
    .. method:: MultiAbstractDeployDevice._get_device_type()

       Private method that evaluates the types of devices managed by this instance. It returns a single type if all devices are of the same type or "mixed_device" if they differ.

       :return: A string indicating the device type.

    .. method:: MultiAbstractDeployDevice.check_valid_device_type(device)

       Validates that a given device's type matches the type expected by this multi-device manager.

       :param device: The device to check.
       :raises AssertionError: If the device type does not match the multi-device type.

    .. method:: MultiAbstractDeployDevice.exist_device(device)

       Checks if a given device is part of the managed devices.

       :param device: The device to check.
       :return: True if the device is managed by this instance, False otherwise.

    Examples
    --------
    Creating an instance of ``MultiAbstractDeployDevice`` with mixed device types:

    .. code-block:: python

        device1 = DeployDevice(device_type='sensor')
        device2 = DeployDevice(device_type='actuator')
        multi_device = MultiAbstractDeployDevice(devices=[device1, device2], strict=False)
        print(multi_device.device_type)  # Outputs: "multi_device: mixed_device"

        # Checking if a device is valid
        try:
            multi_device.check_valid_device_type(device1)
        except AssertionError as e:
            print(e)

    Notes
    -----
    - The class is particularly useful in systems where multiple devices need to be managed collectively, and it is crucial to ensure that operations across these devices are compatible.
    - The flexibility of the `MultiAbstractDeployDevice` allows it to adapt to environments with either homogeneous or heterogeneous device types.

    """
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

    def check_valid_device_type(self, device):
        assert device.device_type == self.device_type, \
            "The device type is not the same as the multi_device type"

    def exist_device(self, device):
        return device in self.devices


class CPUDeployDevice(DeployDevice):
    """
    CPUDeployDevice Class
    =====================

    The ``CPUDeployDevice`` class is a specialized subclass of the ``DeployDevice`` class, specifically designed for deployments that target CPU devices.

    Attributes
    ----------
    device_name : str
        The name of the device, statically set to "cpu" to indicate that this device is a CPU.
    device_type : str
        The type of the device, statically set to "cpu" to clearly define the device's nature as a CPU.

    Constructor
    -----------
    .. method:: CPUDeployDevice.__init__()

       Initializes a new instance of the ``CPUDeployDevice`` class. Since all attributes are statically defined, this constructor only calls the constructor of the superclass, ``DeployDevice``, which sets up any additional provided metadata.

       This class does not take any parameters beyond what ``DeployDevice`` might accept through keyword arguments for additional metadata.

    Example
    -------
    Creating an instance of ``CPUDeployDevice``:

    .. code-block:: python

        cpu_device = CPUDeployDevice()
        print(cpu_device.device_name)  # Outputs: "cpu"
        print(cpu_device.device_type)  # Outputs: "cpu"

    Notes
    -----
    - The ``CPUDeployDevice`` class is particularly useful in scenarios where specific configurations or operations are tied to CPU-based deployments.
    - This class can be extended to include more specific CPU-related configurations and operations if required.

    """
    device_name = "cpu"
    device_type = "cpu"


class GPUDeployDevice(DeployDevice):
    """
    GPUDeployDevice Class
    =====================

    The ``GPUDeployDevice`` class extends the ``DeployDevice`` class, specifically designed for deployments targeting GPU devices. It allows for flexible identification of GPUs through a dynamic naming convention based on the provided device ID.

    Attributes
    ----------
    device_name : str
        The name of the device, dynamically set based on the provided `device_id` during initialization. This can be a specific GPU identifier.
    device_type : str
        The type of the device, statically set to "gpu" to specify the device's nature as a GPU.

    Constructor
    -----------
    .. method:: GPUDeployDevice.__init__(device_id: Union[int, str], **kwargs)

       Initializes a new instance of the ``GPUDeployDevice`` class, setting the GPU device's name based on an identifier and allowing for additional device metadata.

       :param device_id: An integer or string that uniquely identifies the GPU. If an integer is provided, the `device_name` is set to "gpu:{device_id}". If a string is provided, it is used directly as the `device_name`.
       :param kwargs: Arbitrary keyword arguments that are passed to the base class constructor for setting additional metadata.

    Examples
    --------
    Creating an instance of ``GPUDeployDevice`` with an integer device ID:

    .. code-block:: python

        gpu_device = GPUDeployDevice(device_id=0)
        print(gpu_device.device_name)  # Outputs: "gpu:0"
        print(gpu_device.device_type)  # Outputs: "gpu"

    Creating an instance of ``GPUDeployDevice`` with a string device ID:

    .. code-block:: python

        gpu_device = GPUDeployDevice(device_id="NVIDIA GTX 1080")
        print(gpu_device.device_name)  # Outputs: "NVIDIA GTX 1080"
        print(gpu_device.device_type)  # Outputs: "gpu"

    Notes
    -----
    - The ``GPUDeployDevice`` class is particularly useful in systems where GPU devices need distinct identification and management, supporting both numerical and descriptive identifiers.
    - This class can be extended to include more specific GPU-related configurations and operations if required.

    """
    device_name = "gpu"
    device_type = "gpu"

    def __init__(self, device_id: Union[int, str], **kwargs):
        super().__init__(**kwargs)
        if isinstance(device_id, str):
            self.device_name = device_id
        else:
            self.device_name = f"gpu:{device_id}"


def register_deploy_device(device: DeployDevice):
    """
    register_deploy_device Function
    ===============================

    The ``register_deploy_device`` function is used to register an instance of ``DeployDevice`` or its subclasses in a global registry. This allows for centralized management and access to various device deployment configurations.

    Function Definition
    -------------------
    .. function:: register_deploy_device(device: DeployDevice)

       Registers a device deployment configuration by adding it to the ``registered_deploy_device`` dictionary using the device's name as the key.

       :param device: An instance of ``DeployDevice`` or its subclass that should be registered.
       :return: Returns the registered device for possible further use or verification.

    Example
    -------
    Registering a new GPU device:

    .. code-block:: python

        new_gpu_device = GPUDeployDevice(device_id=1)
        registered_device = register_deploy_device(new_gpu_device)
        print(registered_device.device_name)  # Outputs: "gpu:1"
    """
    registered_deploy_device[device.device_name] = device
    return device


registered_deploy_device = {
    "cpu": CPUDeployDevice(),
    "gpu:0": GPUDeployDevice(0)
}
registered_deploy_device.__doc__ = """
registered_deploy_device Dictionary
===================================

A global dictionary that holds references to registered device deployment configurations. Each key is a device name, and the corresponding value is an instance of ``DeployDevice`` or its subclass.

Initial Entries
---------------
The dictionary is initially populated with a CPU device and a GPU device:

.. code-block:: python

    registered_deploy_device = {
        "cpu": CPUDeployDevice(),
        "gpu:0": GPUDeployDevice(0)
    }

Usage
-----
Accessing a registered device:

.. code-block:: python

    cpu_device = registered_deploy_device["cpu"]
    print(cpu_device.device_name)  # Outputs: "cpu"

Adding a new device:

.. code-block:: python

    registered_deploy_device["gpu:1"] = GPUDeployDevice(device_id=1)
    print(registered_deploy_device["gpu:1"].device_name)  # Outputs: "gpu:1"

Notes
-----
- The ``register_deploy_device`` function and the ``registered_deploy_device`` dictionary facilitate the tracking and management of various deployment devices in a unified manner.
- This approach allows for easy access and management of device configurations across different parts of a system."""
