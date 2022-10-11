import os
import pip

try:
    from mercury import mercury
except (ImportError, ModuleNotFoundError):
    # pip install mljar-mercury
    pip.main(['install', 'mljar-mercury'])
    exit("Please run this script again.")

NOTEBOOK_FULL_PATH = "C:/yoni/final_project/mercury_demo/mercury_demo.ipynb"


def main():
    """Opens Mercury demo in your browser."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    url = "http://127.0.0.1:8000/"

    os.system(f"mercury add {NOTEBOOK_FULL_PATH}")

    print(f"Please enter this URL in your browser: {url}")
    print("You might need to wait a few seconds for the server to start.")
    print("Press Ctrl+C (for windows and linux) or command+C (for macOS) to stop the server.")

    os.system(f"mercury run {NOTEBOOK_FULL_PATH}")


if __name__ == "__main__":
    main()
