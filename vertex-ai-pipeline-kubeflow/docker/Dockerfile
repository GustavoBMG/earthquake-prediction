# Specifies base image and tag
# FROM python:3.7
FROM europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest

# WORKDIR /root

# # Installs additional packages
# RUN pip install joblib pandas matplotlib scikit-learn xgboost tensorflow shap
RUN pip3 install pandas matplotlib scikit-learn xgboost tensorflow shap

# # Copies the custom code to the docker image.
COPY zucaml/ ./zucaml

# RUN pip install joblib pandas gcsfs

# RUN pip install six google-cloud-storage

ENV PYTHONPATH="/root"