"""A script that publish a new version of the library to PyPI."""

import os

import build
import toml
from twine.commands.upload import upload
from twine.settings import Settings

script_dir = os.path.dirname(os.path.realpath(__file__))


def version_file_name(version: str) -> str:
    """Return the name of the file that contains the version number."""
    return f"grouped_sampling-{version}.tar.gz"


def version_file_path(version: str) -> str:
    """Return the path of the file that contains the version number."""
    return os.path.join(script_dir, "dist", version_file_name(version))


def increase_version():
    """Increase the version number by one."""
    pyproject = toml.load("pyproject.toml")
    old_version = pyproject["project"]["version"]
    major, minor, patch = old_version.split(".")
    patch = int(patch) + 1
    new_version = f"{major}.{minor}.{patch}"
    pyproject["project"]["version"] = new_version
    with open("pyproject.toml", "w") as f:
        toml.dump(pyproject, f)
    return new_version


def decrease_version():
    """Decrease the version number by one."""
    pyproject = toml.load("pyproject.toml")
    old_version = pyproject["project"]["version"]
    major, minor, patch = old_version.split(".")
    patch = int(patch) - 1
    new_version = f"{major}.{minor}.{patch}"
    return new_version


def version_already_exists(new_version: str) -> bool:
    """Check if the new version already exists."""
    return os.path.exists(version_file_path(new_version))


def build_version(new_version: str) -> None:
    """Build the new version."""
    if version_already_exists(new_version):
        return
    build.ProjectBuilder(srcdir=script_dir).build(
        output_directory="dist", distribution="sdist"
    )
    # return only when the build is finished
    while not version_already_exists(new_version):
        pass


def publish_version(new_version: str):
    """Publish the new version to PyPI."""
    upload_settings = Settings(
        username="__token__",
        password=get_pypi_api_token(),
        repository="pypi",
        disable_progress_bar=True,
    )
    upload(
        upload_settings=upload_settings,
        dists=[version_file_path(new_version)],
    )


def get_pypi_api_token() -> str:
    pypi_api_token_path = os.path.join(script_dir, "pypi_api_token.txt")
    if not os.path.exists(pypi_api_token_path):
        return input("Please enter your PyPI API token: ")
    with open(pypi_api_token_path, "r") as f:
        pypi_api_token = f.read().strip()
    return pypi_api_token


def main():
    # run build
    new_version: str = increase_version()
    build_version(new_version=new_version)
    # Publish the new version to PyPI.
    publish_version(new_version=new_version)


if __name__ == "__main__":
    main()
