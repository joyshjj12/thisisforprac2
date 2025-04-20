FROM python:3-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install -r requirements.txt


WORKDIR /app
COPY . /app

COPY B5_diabetes.csv /app/

RUN python model_pred.py

ENTRYPOINT ["python", "prediction.py"]

CMD []