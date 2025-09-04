FROM python:3.11.6-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ADD src src

# Create logs directory
RUN mkdir -p logs

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

EXPOSE 5000

CMD ["python", "src"]
