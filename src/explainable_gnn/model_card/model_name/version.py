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
    def __init__(self, version):
        """
        :param version: str
                        valid version string:
                            {number}
                            {number}.{number}.{number}
                            {number}.{number}.{number}-{type}
                            {number}.{number}.{number}-{type}{number}
                            {number}.{number}.{number}-{type}{number}.{number}.{number}
        """
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
