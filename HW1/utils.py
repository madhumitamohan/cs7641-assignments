import pickle
from pip._internal.operations.freeze import freeze


class PackageUtils:
    def __init__(self):
        pass

    def get_packages(self=None):
        """
        Do not change. Gets a list of importable packages in the environment.
        """
        if not isinstance(self, PackageUtils):
            raise SyntaxError(
                """Improper usage. `get_packages()` is a member method of PackageUtils, not a static method.
You should create an instance of PackageUtils using its constructor `PackageUtils()`, then call `get_packages()` on that instance."""
            )
        package_list = []
        for package in freeze():
            if "@" in package:
                line = package.split("@")
            elif "==" in package:
                line = package.split("==")
            package_list.append(line[0])
            print(package)
        with open("env.pkl", "wb") as f:
            pickle.dump(package_list, f)
