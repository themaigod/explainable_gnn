from typing import List, Union


class DeployFramework:
    """
    DeployFramework Class
    =====================

    The ``DeployFramework`` class serves as a base class for managing deployment settings of various computing frameworks. It allows for the flexible definition of metadata associated with a specific framework.

    Attributes
    ----------
    framework_name : str or None
        The name of the framework, which can be set dynamically during instantiation or left as `None` to default to "default_framework".

    Constructor
    -----------
    .. method:: DeployFramework.__init__(**kwargs)

       Initializes a new instance of the ``DeployFramework``, optionally setting framework metadata.

       :param kwargs: Arbitrary keyword arguments representing metadata about the framework. These are stored in a dictionary attribute 'framework_meta'.

       If 'framework_name' is not explicitly set during instantiation, it defaults to "default_framework".

    Attributes
    ----------
    framework_meta : dict
        A dictionary to store metadata about the framework, which can include version, configurations, dependencies, etc.

    Examples
    --------
    Creating an instance of ``DeployFramework`` without specifying a framework name:

    .. code-block:: python

        default_framework = DeployFramework()
        print(default_framework.framework_name)  # Outputs: "default_framework"

    PyTorchDeployFramework Class
    ============================

    A subclass of ``DeployFramework`` that specifically handles deployment settings for the PyTorch framework.

    Attributes
    ----------
    framework_name : str
        The name of the framework, statically set to "pytorch".

    Constructor
    -----------
    Inherits the constructor from ``DeployFramework`` and does not add any additional functionality. All attributes and methods are inherited directly.

    Examples
    --------
    Creating an instance of ``PyTorchDeployFramework``:

    .. code-block:: python

        pytorch_framework = PyTorchDeployFramework()
        print(pytorch_framework.framework_name)  # Outputs: "pytorch"

    NumPyDeployFramework Class
    ==========================

    A subclass of ``DeployFramework`` that specifically handles deployment settings for the NumPy framework.

    Attributes
    ----------
    framework_name : str
        The name of the framework, statically set to "numpy".

    Constructor
    -----------
    Inherits the constructor from ``DeployFramework`` and does not add any additional functionality. All attributes and methods are inherited directly.

    Examples
    --------
    Creating an instance of ``NumPyDeployFramework``:

    .. code-block:: python

        numpy_framework = NumPyDeployFramework()
        print(numpy_framework.framework_name)  # Outputs: "numpy"

    Notes
    -----
    - The ``DeployFramework`` class and its subclasses allow for a standardized approach to managing different computing frameworks within a system.
    - These classes can be extended to include more specific configurations or functionality as required by different frameworks.

    """
    framework_name = None

    def __init__(self, **kwargs):
        self.framework_meta = {}
        if kwargs:
            self.framework_meta.update(kwargs)
        if not self.framework_name:
            self.framework_name = "default_framework"


class PyTorchDeployFramework(DeployFramework):
    """
        PyTorchDeployFramework Class
        ============================

        A subclass of ``DeployFramework`` that specifically handles deployment settings for the PyTorch framework.

        Attributes
        ----------
        framework_name : str
            The name of the framework, statically set to "pytorch".

        Constructor
        -----------
        Inherits the constructor from ``DeployFramework`` and does not add any additional functionality. All attributes and methods are inherited directly.

        Examples
        --------
        Creating an instance of ``PyTorchDeployFramework``:

        .. code-block:: python

            pytorch_framework = PyTorchDeployFramework()
            print(pytorch_framework.framework_name)  # Outputs: "pytorch"

    """
    framework_name = "pytorch"


class NumPyDeployFramework(DeployFramework):
    """
            NumPyDeployFramework Class
        ==========================

        A subclass of ``DeployFramework`` that specifically handles deployment settings for the NumPy framework.

        Attributes
        ----------
        framework_name : str
            The name of the framework, statically set to "numpy".

        Constructor
        -----------
        Inherits the constructor from ``DeployFramework`` and does not add any additional functionality. All attributes and methods are inherited directly.

        Examples
        --------
        Creating an instance of ``NumPyDeployFramework``:

        .. code-block:: python

            numpy_framework = NumPyDeployFramework()
            print(numpy_framework.framework_name)  # Outputs: "numpy"
    """
    framework_name = "numpy"


class MultiAbstractDeployFramework(DeployFramework):
    """
    MultiAbstractDeployFramework Class
    ===================================

    The ``MultiAbstractDeployFramework`` class extends the ``DeployFramework`` to manage multiple computing frameworks simultaneously. It is designed to handle complex scenarios where integration across various technologies is required.

    Attributes
    ----------
    framework_name : str
        The name of the framework manager, statically set to "multi_framework".

    Constructor
    -----------
    .. method:: MultiAbstractDeployFramework.__init__(frameworks: List[DeployFramework], **kwargs)

       Initializes a new instance of ``MultiAbstractDeployFramework``, setting up management for multiple frameworks.

       :param frameworks: A list of ``DeployFramework`` instances that this manager will handle.
       :param kwargs: Arbitrary keyword arguments that are passed to the base class constructor for additional metadata setup.

       Post initialization, the constructor calls ``check_valid_framework_type`` to validate the types of frameworks being managed based on the `strict` setting in `framework_meta`.

    Methods
    -------
    .. method:: MultiAbstractDeployFramework.check_valid_framework_type(framework: Union[DeployFramework, List[DeployFramework]])

       Validates that each framework in the provided list or a single framework is registered under this multi-framework manager. Depending on the `strict` mode in `framework_meta`, it either ensures all frameworks are already registered or attempts to register any new frameworks.

       :param framework: A single ``DeployFramework`` instance or a list of them to validate or register.
       :raises ValueError: If `strict` is True and any framework is not a registered `DeployFramework` or not part of the managed frameworks.

    .. method:: MultiAbstractDeployFramework.exist_framework(framework)

       Checks if a given framework is part of the managed frameworks.

       :param framework: The ``DeployFramework`` instance to check.
       :return: True if the framework is managed by this instance, False otherwise.

    Examples
    --------
    Creating an instance of ``MultiAbstractDeployFramework`` with frameworks:

    .. code-block:: python

        pytorch_framework = PyTorchDeployFramework()
        numpy_framework = NumPyDeployFramework()
        multi_framework = MultiAbstractDeployFramework(frameworks=[pytorch_framework, numpy_framework])
        print(multi_framework.exist_framework(pytorch_framework))  # Outputs: True

    Notes
    -----
    - The ``MultiAbstractDeployFramework`` class is particularly useful in environments where multiple deployment frameworks need to be managed collectively, allowing for dynamic checks and registrations based on the configuration requirements.
    - This class enhances flexibility and scalability in managing various technology stacks, especially in complex deployment scenarios.

    """
    framework_name = "multi_framework"

    def __init__(self, frameworks: List[DeployFramework], **kwargs):
        self.frameworks = frameworks
        super().__init__(**kwargs)
        self.check_valid_framework_type(frameworks)

    def check_valid_framework_type(self, framework: Union[
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
    """
    register_deploy_framework Function
    ==================================

    The ``register_deploy_framework`` function is designed to register instances of ``DeployFramework`` or its subclasses into a global registry. This allows for centralized management and easy access to deployment framework configurations.

    Function Definition
    -------------------
    .. function:: register_deploy_framework(framework: DeployFramework)

       Registers a framework deployment configuration by adding it to the ``registered_deploy_framework`` dictionary using the framework's name as the key.

       :param framework: An instance of ``DeployFramework`` or its subclass that should be registered.
       :return: Returns the registered framework for possible further use or verification.

    Example
    -------
    Registering a new framework instance:

    .. code-block:: python

        new_framework = PyTorchDeployFramework()
        registered_framework = register_deploy_framework(new_framework)
        print(registered_framework.framework_name)  # Outputs: "pytorch"
    """
    registered_deploy_framework[framework.framework_name] = framework
    return framework


registered_deploy_framework = {
    "pytorch": PyTorchDeployFramework,
    "numpy": NumPyDeployFramework
}
registered_deploy_framework.__doc__ = """
    registered_deploy_framework Dictionary
    =======================================
    
    A global dictionary that holds references to registered deployment framework configurations. Each key is a framework name, and the corresponding value is an instance of ``DeployFramework`` or its subclass.
    
    Initial Entries
    ---------------
    The dictionary is initially populated with entries for PyTorch and NumPy frameworks:
    
    .. code-block:: python
    
        registered_deploy_framework = {
            "pytorch": PyTorchDeployFramework(),
            "numpy": NumPyDeployFramework()
        }
    
    Usage
    -----
    Accessing a registered framework:
    
    .. code-block:: python
    
        pytorch_framework = registered_deploy_framework["pytorch"]
        print(pytorch_framework.framework_name)  # Outputs: "pytorch"
    
    Adding a new framework:
    
    .. code-block:: python
    
        tensorflow_framework = TensorFlowDeployFramework()
        registered_deploy_framework["tensorflow"] = tensorflow_framework
        print(registered_deploy_framework["tensorflow"].framework_name)  # Outputs: "tensorflow"
    
    Notes
    -----
    - The ``register_deploy_framework`` function and the ``registered_deploy_framework`` dictionary facilitate the tracking and management of various deployment frameworks in a unified manner.
    - This approach allows for easy access and management of framework configurations across different parts of a system.
    """
