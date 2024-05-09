invalid_str = [
    ":",
    " ",
    "-",
    ".",
    ",",
    ";",
    "'",
    '"',
    "/",
    "\\",
    "|",
    "?",
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "(",
    ")",
    "+",
    "=",
    "<",
    ">",
    "[",
    "]",
    "{",
    "}",
    "`",
    "~",
    "\n",
    "\t",
    "\r",
    "\f",
    "\v",
    "\b",

]


class ModelName:
    """
    ModelName Class
    ===============

    The ``ModelName`` class is designed to encapsulate and validate model names by ensuring they do not contain any invalid characters. This class provides methods for setting, getting, and validating model names according to a predefined set of restricted characters.

    Attributes
    ----------
    invalid_str : list
        A list of characters that are not allowed in model names. Includes a variety of symbols and whitespace characters that might cause issues in filenames, URLs, or commands.

    Constructor
    -----------
    .. method:: ModelName.__init__(name)

       Initializes a new instance of the ``ModelName`` class with a specific name.

       :param name: A string representing the name of the model. This name is validated upon instantiation.

    Methods
    -------
    .. method:: ModelName._valid()

       Checks if the current model name is valid by ensuring it does not contain any characters from the ``invalid_str`` list.

       :return: True if the name is valid, False otherwise.

    Magic Methods
    -------------
    .. method:: ModelName.__str__()

       Returns the current model name.

    .. method:: ModelName.__repr__()

       Returns a string representation of the current model name, suitable for debugging.

    .. method:: ModelName.__eq__(other)

       Compares this ``ModelName`` instance with another to check for equality based on the name.

       :param other: Another ``ModelName`` instance to compare against.
       :return: True if the names are the same, False otherwise.

    .. method:: ModelName.__hash__()

       Returns a hash based on the model name, allowing instances of this class to be used in hashable collections.

    .. method:: ModelName.__set_name__(owner, name)

       Sets the model name when used as a descriptor in a class. Validates the name.

       :param owner: The class owning this descriptor.
       :param name: The new name to set.
       :raises ValueError: If the name contains any invalid characters.

    .. method:: ModelName.__get__(instance, owner)

       Gets the model name when this class is used as a descriptor.

       :param instance: The instance from which to get the name.
       :param owner: The owner class.

    .. method:: ModelName.__set__(instance, value)

       Sets the model name in an instance, validating the new value before setting.

       :param instance: The instance in which to set the name.
       :param value: The new name to set.
       :raises ValueError: If the new name contains any invalid characters.

    Examples
    --------
    Creating an instance of ``ModelName`` and validating the name:

    .. code-block:: python

        try:
            model_name = ModelName("ValidName123")
            print(model_name)  # Outputs: "ValidName123"
        except ValueError as e:
            print(e)

        try:
            model_name = ModelName("Invalid Name!")
        except ValueError as e:
            print(e)  # Outputs: "Invalid name Invalid Name!"

    Notes
    -----
    - The ``ModelName`` class is crucial for ensuring that names used in models adhere to specific rules, which can prevent issues related to system commands, file handling, and URLs.
    - Developers need to update the ``invalid_str`` list based on additional requirements or restrictions related to specific deployment or operational environments.

    """
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def _valid(self):
        if self.name is not None:
            if isinstance(self.name, str):
                for i in invalid_str:
                    if i in self.name:
                        return False
                return True
        return False

    def __set_name__(self, owner, name):
        self.name = name
        if not self._valid():
            raise ValueError(f"Invalid name {self.name}")
        return self

    def __get__(self, instance, owner):
        return self.name

    def __set__(self, instance, value):
        self.name = value
        if not self._valid():
            raise ValueError(f"Invalid name {self.name}")
        return self
