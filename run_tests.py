import os
import sys

import pytest


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root + "/src")
    sys.path.append(project_root + "/tests")
    pytest.main(["tests"])


if __name__ == "__main__":
    main()
