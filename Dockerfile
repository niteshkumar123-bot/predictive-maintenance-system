FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build-time model generation makes a fresh container self-contained.
RUN python run_pipeline.py

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
