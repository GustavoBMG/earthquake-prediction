#!/bin/bash

export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export REPO_NAME=earhquake-prediction
export IMAGE_NAME=serving-xgboost
export IMAGE_TAG=basic
export IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

export 

docker build -f Dockerfile -t ${IMAGE_URI} ./

gcloud auth configure-docker us-central1-docker.pkg.dev

docker push ${IMAGE_URI}