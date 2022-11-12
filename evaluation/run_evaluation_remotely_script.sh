rm -rf ./final_project
git clone --quiet https://github.com/yonikremer/final_project.git
pip install --quiet --requirement ./final_project/evaluation/evaluation_requirements.txt
pip install --quiet --requirement ./final_project/project_requirements.txt
export PYTHONPATH="${PYTHONPATH}:/kaggle/working/final_project" && python ./final_project/evaluation/evaluate_translation.py