import os
import sys

import pytest
from _pytest.config import ExitCode
from coverage import Coverage

script_dir = os.path.dirname(os.path.realpath(__file__))


def main() -> bool:
    """Run all tests and return True if all tests passed, False otherwise."""
    excluded = [
        "*/tests/*",
        "*/__init__.py",
        os.path.join(script_dir, "web_app", "*"),
        "*raise*",
    ]
    included = os.path.join(script_dir, "src", "*")
    cov = Coverage(
        omit=excluded,
        include=included,
    )
    cov.start()

    exitcode: ExitCode = pytest.main([
        os.path.join(script_dir, "tests"), "--verbose", "--color=yes",
        "--tb=short"
    ])

    cov.stop()
    cov.save()
    report = cov.report(
        show_missing=True,
        skip_covered=True,
        omit=[
            "*/tests/*", "*/__init__.py",
            os.path.join(script_dir, "web_app", "*")
        ],
        skip_empty=True,
        include=[os.path.join(script_dir, "src", "*")],
    )
    cov.html_report(directory=os.path.join(script_dir, "coverage_html_report"))
    return exitcode == ExitCode.OK and report in (1, 100)


if __name__ == "__main__":
    all_tests_passed = main()
    if all_tests_passed:  # if all tests passed
        sys.exit(0)
    else:
        sys.exit(1)  # if any test failed / error occurred / etc.
