# Use a base Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y default-jdk-headless procps && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file initially
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (this will be cached if requirements.txt doesn't change)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install langchain

RUN pip install --no-cache-dir -r /app/requirements.txt --timeout=600

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8082"]