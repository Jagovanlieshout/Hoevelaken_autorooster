# Use a lightweight Python image
FROM python:3.11-slim

# Set environment variables (no .pyc, ensure output flush)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory inside container
WORKDIR /app

# Install system dependencies if needed (optional)
# RUN apt-get update && apt-get install -y build-essential

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port Flask runs on
EXPOSE 8000

# Run Flask with Gunicorn (production ready)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]