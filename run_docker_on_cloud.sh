# This is a sequence of commands that connects to the cloud server and runs the script run_benchmarks.py

# move to the Home directory of ubuntu
cd .. && cd ..

# connect to the cloud server, replace IP_ADDRESS with the IP address of the server
sudo ssh -i Downloads/yonikey.pem ubuntu@IP_ADDRESS

# run the docker image
sudo docker run --gpus all -it yonikremer/human_eval:latest python3 -m human_eval_project.human_eval.main