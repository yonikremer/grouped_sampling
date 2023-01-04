# Grouped Text Generation

My high school research project.
## The research question:
> Can we use generate n tokens using less than n calls to a causal language model with decoder only transformer architecture?
> If yes, How? How the text generated in this way different from the text generated using n calls?
> If no, Why?
> I show that such an algorithm is possible,
> and the text generated using this algorithm is different from the text generated using n calls.
> I will also so significant improvement in the runtime of my algorithm.

# Project OverView:

## The project paper:
Is written in Hebrew in [documents\העבודה עצמה.docx](documents\העבודה%20עצמה.docx)

## The code:

### Before running any of the code:
Please make sure U have python 3 (3.9 is recommended) installed in your system and Internet connection.
install the requirements using `python -m pip install -r requirements.txt`

### The Algorithm Itself:
Is implemented in the 'text_generator.py', 'sampling_generator.py' and 'tree_generator.py' files.
To use it:
import sampling_generator\tree_generator
create SamplingGenerator\TreeGenerator object and call the '__call__' method.

### The Web App:
Is implemented in the web_app folder.
To use it:
install the requirements using `python -m pip install -r requirements.txt` and `python -m pip install -r web_app/web_app_requirements.txt`
run the app locally using `python open_web_app.py`

### The demos
Are a simple user interfaces for the algorithm.
to run the demo on colab:
enter [This Link](https://colab.research.google.com/github/yonikremer/final_project/blob/master/colab_demo.ipynb) and run all the cells.
to run it locally:
install the requirements using `python -m pip install -r \mercury_demo\mercury_demo_requirements.txt`
run the demo using `python open_mercury_demo.py`

### Evaluation:
As of today, the evaluation is only checking the [bertscore](https://arxiv.org/abs/1904.09675) on the [news commentary dataset](https://opus.nlpl.eu/News-Commentary.php).
More evaluation options will be added in the future.
Is implemented in the evaluation folder.
To evaluate the algorithm:
make sure you have an account in [comet machine learning](https://www.comet.com/site/)
create a project called 'grouped-sampling-evaluation'
copy your api key to the 'evaluation\comet_api_key.txt' file
install the requirements using `python -m pip install -r evaluation/evaluation_requirements.txt`
run the evaluation using `python evaluation\evaluate_translation.py`

### Training Causal Language Models:
The code for training the NORMAL causal language models is in the 'model' folder.
