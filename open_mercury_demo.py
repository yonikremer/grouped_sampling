import os
import pip
import webbrowser

try:
    from mercury import mercury
except (ImportError, ModuleNotFoundError):
    # pip install mljar-mercury
    pip.main(['install', 'mljar-mercury'])
    exit("Please run this script again.")


def main():
    """Opens Mercury demo in your browser."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    url = "http://127.0.0.1:8000/"
    print(f"Please enter this URL in your browser: {url}")
    os.system("mercury run mercury_demo.ipynb")


if __name__ == "__main__":
    main()
