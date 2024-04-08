from .name import ModelName
from .version import ModelVersion


def _split_name_version(name):
    name = name.strip()
    if ':' in name:
        name, version = name.split(':')
        return name, version
    return name, None


class ModelFullName:
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
