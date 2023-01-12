# Grouped Sampling - Efficient Use Of Causal Language Models

## My high school research project in computer science.

# The research question:
> Can we use generate n tokens using less than n calls to a Causal Language model?
> If yes, How? 
> How the text generated in this way differs from the text generated using n calls?
> If no, Why?


# The Answer:

You CAN generate n tokens with less than n calls to A Causal Language Model.

In fact, You can generate long texts with a single call to A Causal Language Model.

The grouped sampling algorithm is more efficient than all existing algorithms.

The algorithm succeeded more than the existing alternative in the experiments.

# Using Grouped Sampling:

1. Make sure You have python 3 (3.9+ is recommended).
2. Make sure You have a fast internet connection
3. Run: `python -m pip install -q grouped-sampling`
4. `from grouped_sampling.sampling_generator import GroupedSamplingPipeLine`
5. Choose a Causal Language Model from huggingface hub
6. Choose a group size, 
The recommended group size is the upper limit for the length of the generated texts.
A higher group size will cause unnecessary computations.
A lower group size will cause lower performance both in runtime and text quality.
7. `pipe = GroupedSamplingPipeLine(model_name=YOUR_MODEL_NAME, group_size=YOUR_GROUP_SIZE)`
8. `answer = pipe(YOUR_TEXT)["generated_text"]`


# Project OverView:

## The project book:
The project book is where I explain the project in great details.
The book includes a detailed explanation of the technical background and the solution. 
It is written in Hebrew in the file [documents\העבודה עצמה.docx](documents\העבודה%20עצמה.docx)

## Other Documents:
There are other documents I created for the project
All of them are in Hebrew
All are available in the documents folder
1. A presentation.
2. A poster.
3. A project summary.

## The code:

### The Algorithm Itself:
You can install it using `python -m pip install -q grouped-sampling`
All of the code is under the src/grouped_sampling

### The Demo
Are a simple user interfaces for the algorithm.
To run the demo on colab:
enter [This Link](https://colab.research.google.com/github/yonikremer/final_project/blob/master/colab_demo.ipynb) and run all the cells.
