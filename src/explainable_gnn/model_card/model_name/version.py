invalid_str = [
    ":",
    " ",
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


class ModelVersion:
    """
    ModelVersion Class
    ==================

    The ``ModelVersion`` class is designed to encapsulate and validate version strings for models, ensuring they conform to specific formatting standards. This class incorporates rigorous checks to validate the format of version strings, adhering to software versioning standards.

    Constructor
    -----------
    .. method:: ModelVersion.__init__(version)

       Initializes a new instance of the ``ModelVersion`` class with a specified version string. The version string is validated against a defined format upon initialization.

       :param version: A string representing the version of the model. It must adhere to one of the following formats:
                       - Purely numeric (e.g., "1", "2.1")
                       - Dot-separated numbers (e.g., "1.0.0")
                       - Numbers followed by a pre-release type and optional additional numbers (e.g., "1.0.0-alpha", "1.0.0-beta1", "1.0.0-beta1.2.3")
       :raises ValueError: If the version string contains invalid characters, does not start with a digit, or does not follow the formatting rules.

    Attributes
    ----------
    version : str
        Stores the validated version string of the model.

    Methods
    -------
    .. method:: ModelVersion._valid()

       Validates the format of the version string. This private method checks the string against a series of format rules to ensure compliance with standard versioning practices.

       :return: True if the version string is valid, False otherwise.

    .. method:: ModelVersion.compare(other)

         Compares this version with another version to determine their relative order based on the numeric components of the version strings.

        :param other: Another ``ModelVersion`` instance to compare against.
        :return: 0 if the versions are equal, 1 if this version is greater, -1 if this version is less.

    Magic Methods
    -------------
    .. method:: ModelVersion.__str__()

       Returns the version string.

    .. method:: ModelVersion.__repr__()

       Returns a string representation of the version, suitable for debugging.

    .. method:: ModelVersion.__eq__(other)

       Compares this ``ModelVersion`` instance with another to check for equality based on the entire version string.

       :param other: Another ``ModelVersion`` instance.
       :return: True if the entire version strings are identical, False otherwise.

    .. method:: ModelVersion.__hash__()

       Returns a hash based on the version string.

    Version Comparison Methods
    --------------------------
    These comparison methods only consider the numeric parts of the version strings, ignoring any pre-release or build metadata.

    .. method:: ModelVersion.__lt__(other)

       Checks if this version is less than another version based on their numeric components.

    .. method:: ModelVersion.__le__(other)

       Checks if this version is less than or equal to another version based on their numeric components.

    .. method:: ModelVersion.__ne__(other)

       Checks if this version is not equal to another based on their numeric components.

    .. method:: ModelVersion.__gt__(other)

       Checks if this version is greater than another version based on their numeric components.

    .. method:: ModelVersion.__ge__(other)

       Checks if this version is greater than or equal to another version based on their numeric components.

    Examples
    --------
    Creating and using a ``ModelVersion``:

    .. code-block:: python

        try:
            version = ModelVersion("1.0.0-beta1")
            print(version)  # Outputs: "1.0.0-beta1"
        except ValueError as e:
            print(e)  # E.g., "Invalid version string"

        version1 = ModelVersion("1.0.0-alpha")
        version2 = ModelVersion("1.0.0-beta")
        print(version1 < version2)  # Outputs: False, comparisons ignore non-numeric parts

    Notes
    -----
    - The ``ModelVersion`` class plays a crucial role in ensuring that version strings are correctly formatted and validated, particularly in environments where software versioning impacts compatibility and deployment decisions.
    - The distinct behaviors of the equality and comparison methods need careful consideration when used to ensure that comparisons align with the intended use cases of versioning in a software environment.

    """

    def __init__(self, version):
        self.version = version
        self._valid()

    def __str__(self):
        return self.version

    def __repr__(self):
        return self.version

    def __eq__(self, other):
        return self.version == other.version

    def __hash__(self):
        return hash(self.version)

    def _valid(self):
        if self.version is None:
            raise ValueError("Version cannot be None")
        if not isinstance(self.version, str):
            raise ValueError(f"Version must be a string, not {type(self.version)}")
        if len(self.version) == 0:
            raise ValueError("Version cannot be an empty string")
        if len(self.version) > 100:
            raise ValueError("Version cannot be more than 100 characters long")
        if any([char in self.version for char in invalid_str]):
            raise ValueError(
                f"Version cannot contain any of the following characters: {invalid_str}")
        if not self.version[0].isdigit():
            raise ValueError("Version must start with a digit")
        split_version = self.version.split("-")
        if len(split_version) > 2:
            raise ValueError("Version cannot have more than one '-'")

        def valid_number(num_str):
            split_num = num_str.split(".")
            if any([not num.isdigit() for num in split_num]):
                return False
            return True

        if len(split_version) == 2:
            if not valid_number(split_version[0]):
                raise ValueError(
                    "Version number must be in the format {number}.{number}.{number}")
            if not split_version[1][0].isalpha():
                raise ValueError("Version type must start with a letter")
            num_str = split_version[1].split(".")
            if len(num_str) > 1:
                num_str = [num_str[0][-1]] + num_str[1:]
                if not valid_number(".".join(num_str)):
                    raise ValueError(
                        "Version number must be in the format {number}.{number}.{number}")
        else:
            if not valid_number(self.version):
                raise ValueError(
                    "Version number must be in the format {number}.{number}.{number}")

        return True

    def __set_name__(self, owner, name):
        self.version = name
        if not self._valid():
            raise ValueError(f"Invalid version {self.version}")

    def __set__(self, instance, value):
        self.version = value
        if not self._valid():
            raise ValueError(f"Invalid version {self.version}")

    def __get__(self, instance, owner):
        return self.version

    def compare(self, other):
        if self.version.split("-")[0] == other.version.split("-")[0]:
            return 0
        self_version = list(map(int, self.version.split("-")[0].split(".")))
        other_version = list(map(int, other.version.split("-")[0].split(".")))
        for i in range(min(len(self_version), len(other_version))):
            if self_version[i] > other_version[i]:
                return 1
            elif self_version[i] < other_version[i]:
                return -1
        if len(self_version) > len(other_version):
            return 1
        return -1

    def __lt__(self, other):
        return self.compare(other) == -1

    def __le__(self, other):
        return self.compare(other) != 1

    def __ne__(self, other):
        return self.compare(other) != 0

    def __gt__(self, other):
        return self.compare(other) == 1

    def __ge__(self, other):
        return self.compare(other) != -1
