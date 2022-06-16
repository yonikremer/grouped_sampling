# final_project

My high school research project is about changes to the Transformer architecture.

To create a new model from the same architecture:

1. Download the [dataset](https://www.kaggle.com/datasets/urbanbricks/wikipedia-promotional-articles) to your google drive
2. Download model/final_project.ipynb to drive
3. Run the imports & installs
4. Restart your notebook
5. Make sure you use GPU/TPU acceleration
6. Run all cells in an order

To use my models (Unfortunately, I don't have my web app hosted currently):

1. Make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed and updated
2. [Add conda to PATH](https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10)
3. Download /webapp to my_path
4. Open **windows cmd** 
5. Type:
  'cd my_path'/n
  'conda create -n my_venv python=3.9'/n
  'conda activate my_venv'/n
  'pip install -q -r requirements.txt'/n
  'set FLASK_APP=flaskr'/n
  'pytest'/n
  'flask init-db'/n
  'flask run'/n
6. Enter http://127.0.0.1:5000/register

If you have any issues with the project, contact me at yoni.kremer@gmail.com
