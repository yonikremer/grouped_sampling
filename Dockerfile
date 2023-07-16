# Start from the base image
FROM huggingface/transformers-pytorch-gpu:latest

# Set the Hugging Face cache location to this directory
ENV TRANSFORMERS_CACHE=/models

# Install bitsandbytes
RUN pip3 install bitsandbytes

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the code into the container
COPY / .

CMD ["python3", "-m", "human_eval_project.human_eval.main"]