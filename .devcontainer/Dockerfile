FROM tensorflow/tensorflow:2.9.1-gpu

RUN apt-get update && apt-get install -y git graphviz
RUN pip install --upgrade pip
RUN mkdir /project
COPY ../requirements.txt /project/requirements.txt
RUN pip install --no-cache-dir -r /project/requirements.txt