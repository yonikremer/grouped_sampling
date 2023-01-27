import os
import sys

import coverage
import pytest
from _pytest.config import ExitCode

script_dir = os.path.dirname(os.path.realpath(__file__))


def main() -> bool:
    """Run all tests and return True if all tests passed, False otherwise."""
    cov = coverage.Coverage()
    cov.start()

    exitcode: ExitCode = pytest.main([script_dir, "--verbose", "--color=yes"])
    cov.stop()
    cov.save()
    report = cov.report()
    return exitcode == ExitCode.OK and report in (1, 100)


if __name__ == "__main__":
    webapp_folder = os.path.join(script_dir, "web_app")
    sys.path.append(webapp_folder)
    if main():  # if all tests passed
        sys.exit(0)
    else:
        sys.exit(1)  # if any test failed / error occurred / etc.
