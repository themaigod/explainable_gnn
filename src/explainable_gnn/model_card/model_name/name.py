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
