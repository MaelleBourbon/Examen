FROM python:3.9-slim

ENV PATH="/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV LANG=C.UTF-8
WORKDIR /app

# dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD python train_classifier.py && python predict_classification.py
