registered_deploy_requirement = {
}


class DeployRequirement:
    requirement_name = "default_requirement"

    def __init__(self, **kwargs):
        self.requirement_meta = {}
        if kwargs:
            self.requirement_meta.update(kwargs)
        if not self.requirement_name:
            self.requirement_name = "default_requirement"


def register_deploy_requirement(requirement: DeployRequirement):
    registered_deploy_requirement[requirement.requirement_name] = requirement
    return requirement
