registered_deploy_requirement = {
}
registered_deploy_requirement.__doc__ = """
    registered_deploy_requirement Dictionary
    ========================================
    
    A global dictionary that holds references to registered deployment requirements. Each key is a requirement name, and the corresponding value is an instance of ``DeployRequirement``.
    
    Usage
    -----
    Accessing a registered requirement:
    
    .. code-block:: python
    
        requirement = registered_deploy_requirement["default_requirement"]
        print(requirement.requirement_name)  # Outputs: "default_requirement"
    
    Adding a new requirement:
    
    .. code-block:: python
    
        registered_deploy_requirement["specific_requirement"] = DeployRequirement(requirement_name="specific_requirement")
        print(registered_deploy_requirement["specific_requirement"].requirement_name)  # Outputs: "specific_requirement"
    
    Notes
    -----
    - The ``register_deploy_requirement`` function and the ``registered_deploy_requirement`` dictionary facilitate the tracking and management of various deployment requirements in a unified manner.
    - This approach allows for easy access and management of requirement configurations across different parts of a system.
    """


class DeployRequirement:
    """
    DeployRequirement Class
    =======================

    The ``DeployRequirement`` class serves as a base class for defining specific deployment requirements. It allows for the flexible specification of metadata associated with a deployment requirement.

    Attributes
    ----------
    requirement_name : str
        The name of the requirement, which defaults to "default_requirement" if not specified during initialization.

    Constructor
    -----------
    .. method:: DeployRequirement.__init__(**kwargs)

       Initializes a new instance of the ``DeployRequirement``, optionally setting requirement metadata.

       :param kwargs: Arbitrary keyword arguments representing metadata about the requirement. These are stored in a dictionary attribute 'requirement_meta'.

       If 'requirement_name' is not explicitly set during instantiation, it defaults to "default_requirement".

    Attributes
    ----------
    requirement_meta : dict
        A dictionary to store metadata about the deployment requirement, which can include details like version requirements, environment specifications, hardware needs, etc.

    Examples
    --------
    Creating an instance of ``DeployRequirement`` without specifying a requirement name:

    .. code-block:: python

        default_requirement = DeployRequirement()
        print(default_requirement.requirement_name)  # Outputs: "default_requirement"
    """
    requirement_name = "default_requirement"

    def __init__(self, **kwargs):
        self.requirement_meta = {}
        if kwargs:
            self.requirement_meta.update(kwargs)
        if not self.requirement_name:
            self.requirement_name = "default_requirement"


def register_deploy_requirement(requirement: DeployRequirement):
    """
    register_deploy_requirement Function
    ====================================

    The ``register_deploy_requirement`` function is designed to register instances of ``DeployRequirement`` into a global registry. This facilitates centralized management and easy access to requirement configurations.

    Function Definition
    -------------------
    .. function:: register_deploy_requirement(requirement: DeployRequirement)

       Registers a deployment requirement by adding it to the ``registered_deploy_requirement`` dictionary using the requirement's name as the key.

       :param requirement: An instance of ``DeployRequirement`` that should be registered.
       :return: Returns the registered requirement for possible further use or verification.

    Example
    -------
    Registering a new requirement instance:

    .. code-block:: python

        new_requirement = DeployRequirement()
        registered_requirement = register_deploy_requirement(new_requirement)
        print(registered_requirement.requirement_name)  # Outputs: "default_requirement"
    """
    registered_deploy_requirement[requirement.requirement_name] = requirement
    return requirement
