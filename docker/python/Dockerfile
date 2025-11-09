# Use the official Python image from the Docker Hub
FROM python:3.12-slim
LABEL Chatbot with a GNN - Mathilde Patrissi Università di Firenze

RUN apt-get update	
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install curl git wget vim unzip

# Set the working directory in the container
WORKDIR /app
# Copy the current directory contents into the container at /app
#COPY ./app/ /app

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install any needed packages specified in requirements.txt
RUN pip3 install -U pip
#RUN pip3 install --no-cache-dir -r requirements.txt


RUN (cd /app && mkdir /app/ai && mkdir /app/ai/script) 
WORKDIR /app/ai/script

#https://raw.githubusercontent.com/mathyptr/XAI/refs/heads/main/script/shap_explain.py
#RUN wget  https://github.com/mathyptr/XAI/archive/refs/heads/shap_explain.py




