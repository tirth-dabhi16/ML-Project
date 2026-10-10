FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt

COPY app ./app
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]