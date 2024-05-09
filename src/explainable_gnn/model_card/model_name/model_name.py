from .name import ModelName
from .version import ModelVersion


def _split_name_version(name):
    """
    _split_name_version Function
    ===========================

    The ``_split_name_version`` function is designed to parse a full model name string, separating it into distinct name and version components. This utility is integral to handling models where a name-version format is expected.

    Function Definition
    -------------------
    .. function:: _split_name_version(name)

       Parses a string that potentially includes a model name followed by a version, separated by a colon (':').

       :param name: A string formatted as "name:version". The version part is optional.
       :return: A tuple (name, version) where:
           - name (str): The model's name extracted from the input string.
           - version (str or None): The version number extracted if present, otherwise None.

       If a colon is present in the string, the text before the colon is treated as the name, and the text after the colon as the version. If no colon is present, the entire string is treated as the name, and the version is returned as None.

    Examples
    --------
    Examples of how to use ``_split_name_version``:

    .. code-block:: python

        name, version = _split_name_version("ModelX:1.0.0")
        print(name)    # Outputs: "ModelX"
        print(version) # Outputs: "1.0.0"

        name, version = _split_name_version("ModelY")
        print(name)    # Outputs: "ModelY"
        print(version) # Outputs: None

    Use Case
    --------
    This function is used within the ``ModelFullName`` class constructor to correctly assign the name and version attributes when an instance is created with a full name that may include a version. It allows the class to flexibly handle strings with or without a specified version.

    .. note::
        It assumes that the ``name`` parameter is a valid string as a ``ModelFullName`` instance's string representation is passed to it.

    This utility function is crucial for ensuring the correct parsing and handling of model name strings that might include version details, especially in systems where naming conventions include versioning directly in the model name.

    """
    name = name.strip()
    if ':' in name:
        name, version = name.split(':')
        return name, version
    return name, None


class ModelFullName:
    """
    ModelFullName Class
    ===================

    The ``ModelFullName`` class is designed to encapsulate and manage a model's full name, combining both its name and version components into a single entity. It leverages the ``ModelName`` and ``ModelVersion`` classes to ensure that both parts conform to specific standards.

    Utility Function
    ----------------
    .. function:: _split_name_version(name)

       Parses a given full name string into separate name and version components based on the presence of a colon (':') delimiter.

       :param name: A string that potentially contains a name and a version, delimited by a colon.
       :return: A tuple containing the name and version as separate strings. If no colon is present, the version part is returned as None.

    Constructor
    -----------
    .. method:: ModelFullName.__init__(name)

       Initializes a new instance of the ``ModelFullName`` class using a combined name and version string.

       :param name: A string that contains both the name and the version of the model, expected to be delimited by a colon.

       The constructor uses the ``_split_name_version`` function to separate the name and version, initializing ``ModelName`` and ``ModelVersion`` objects respectively.

    Magic Methods
    -------------
    .. method:: ModelFullName.__str__()

       Returns the full representation of the model's name and version, concatenating the two with a colon in between.

    .. method:: ModelFullName.__repr__()

       Provides a more formal representation of the model's full name, useful for debugging and logging.

    .. method:: ModelFullName.__eq__(other)

       Determines if two ``ModelFullName`` instances are equal by comparing both their names and versions.

       :param other: Another ``ModelFullName`` instance to compare against.
       :return: True if both the name and version match, False otherwise.

    .. method:: ModelFullName.__hash__()

       Returns a hash based on the combined string representation of the model's full name.

    .. method:: ModelFullName.__set_name__(owner, name)

       Sets the model's full name using a descriptor, validating and splitting the name and version accordingly.

    .. method:: ModelFullName.__set__(instance, value)

       Sets the model's full name on an instance, ensuring the name and version are properly validated and set.

    .. method:: ModelFullName.__get__(instance, owner)

       Retrieves the full name string of the model from an instance.

    Examples
    --------
    Creating and using a ``ModelFullName``:

    .. code-block:: python

        full_name = ModelFullName("ModelX:1.0.0")
        print(full_name)  # Outputs: "ModelX:1.0.0"

        try:
            full_name = ModelFullName("ModelX:beta")
        except ValueError as e:
            print(e)  # Outputs error if the version format is incorrect

    Notes
    -----
    - The ``ModelFullName`` class is crucial for systems where full specification of a model (including its version) is required for tracking, deployment, or version control.
    - This class ensures that the model name and version are consistently formatted and validated, providing a reliable way to manage model identifiers throughout their lifecycle.

    """
    def __init__(self, name):
        name, version = _split_name_version(name)
        self.name = ModelName(name)
        if version is not None:
            self.version = ModelVersion(version)
        else:
            self.version = None

    def __str__(self):
        return str(self.name) + ':' + str(self.version)

    def __repr__(self):
        return str(self.name) + ':' + str(self.version)

    def __eq__(self, other):
        return self.name == other.name and self.version == other.version

    def __hash__(self):
        return hash(str(self))

    def __set_name__(self, owner, name):
        name, version = _split_name_version(name)
        self.name = ModelName(name)
        if version is not None:
            self.version = ModelVersion(version)
        else:
            self.version = None

    def __set__(self, instance, value):
        name, version = _split_name_version(value)
        self.name = ModelName(name)
        if version is not None:
            self.version = ModelVersion(version)
        else:
            self.version = None

    def __get__(self, instance, owner):
        return str(self)
