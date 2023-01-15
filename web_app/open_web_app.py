import os

from flask import Flask

from web_app.flaskr.__init__ import create_app


def main():
    """Starts hosting the flask web application locally in debug mode"""
    this_files_dir = os.path.dirname(os.path.abspath(__file__))
    web_app_path = os.path.join(this_files_dir, "")
    os.chdir(web_app_path)

    app: Flask = create_app()

    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == "__main__":
    main()
