A web app that allows users to use Grouped Sampling to generate texts.
They can control all the parameters such as the number of groups, the number of samples per group, the number of tokens per sample, and the temperature.
The app also allows users to compare the text generated using different parameters and models.

The app isn't deployed yet, but you can run it locally by following the instructions below.
# Running the app locally
## Prerequisites
- Python 3
- Internet
- Git
- A GPU (optional)

## Installation
1. Clone the repository to a directory I'll call `grouped_sampling_dir`.
2. Install the requirements
```!pip install -r grouped_sampling_dir/grouped_sampling/requirements.txt -r grouped_sampling_dir/grouped_sampling/web_app/web_app_requirements.txt```
3. run the app ```!python grouped_sampling_dir/grouped_sampling/web_app/app.py```