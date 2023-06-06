# Dockerfile

# pull the official docker image
FROM python:3.8-slim-buster

# set work directory
WORKDIR /app

COPY requirements.txt .

# install
RUN apt-get update -y && apt-get install build-essential cmake pkg-config libsndfile1-dev -y
RUN apt-get update -y && apt-get install libgl1 -y


# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

#
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]