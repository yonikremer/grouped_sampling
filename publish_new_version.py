"""A script that publish a new version of the library to PyPI."""

import os

import toml
import build
import twine


def increase_version(version: str) -> str:
    """Increase the version number by one."""
    major, minor, patch = version.split(".")
    patch = int(patch) + 1
    return f"{major}.{minor}.{patch}"


def main():
    # Get the current version from the pyproject.toml file.
    pyproject = toml.load("pyproject.toml")
    old_version = pyproject["project"]["version"]
    print(f"Current version: {old_version}")
    new_version = increase_version(old_version)
    print(f"New version: {new_version}")
    pyproject["project"]["version"] = new_version
    with open("pyproject.toml", "w") as f:
        toml.dump(pyproject, f)

    # cd into the src/grouped_sampling directory.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(script_dir, "src", "grouped_sampling"))

    # run build
    build.ProjectBuilder().build()

    # cd back to the root directory.
    os.chdir(script_dir)

    # Publish the new version to PyPI.
    files_to_publish = f"dist/grouped_sampling-{new_version}.tar.gz, " \
                       f"dist/grouped_sampling-{new_version}-py3-none-any.whl"
    username = "__token__"
    pypi_api_token_path = os.path.join(script_dir, "pypi_api_token.txt")
    with open(pypi_api_token_path, "r") as f:
        pypi_api_token = f.read().strip()
    subprocess.call(["twine", "upload", "--username", username, "--password", pypi_api_token, files_to_publish])


if __name__ == "__main__":
    main()
