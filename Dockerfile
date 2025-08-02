# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed by numpy, opencv, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and related tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirement list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app into container
COPY . .

# Expose the port (Render expects 0.0.0.0:5000)
EXPOSE 5000

# Start the server with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
