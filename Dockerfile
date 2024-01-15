# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        gcc \
        g++ \
        libgl1-mesa-glx \
        libglib2.0-0 \
        procps \
        && rm -rf /var/lib/apt/lists/*

# Install required python packages
RUN pip install numpy pandas tqdm opencv-python torch torchvision python-multipart rembg
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install fastapi uvicorn

# Make port available to the world outside this container
EXPOSE 5001

# Run uvicorn when the container launches
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5001"]