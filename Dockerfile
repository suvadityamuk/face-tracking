FROM python:3.12-bullseye

RUN apt-get update

WORKDIR /workspaces/face-tracking

COPY . /workspaces/face-tracking

RUN pip3 install -r requirements.txt