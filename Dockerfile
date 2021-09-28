FROM python:3-slim

RUN apt update && apt install -y libomp-dev
ADD requirements.txt .
RUN pip install --no-cache -r requirements.txt