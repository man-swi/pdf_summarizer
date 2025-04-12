# Dockerfile (No major changes needed for quantization dependencies if requirements.txt is correct)

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed BEFORE pip installs
# Includes Tesseract and libraries needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Add any other system libraries if needed (unlikely for bitsandbytes/accelerate)
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the requirements file first (Leverages Docker cache)
COPY requirements.txt .

# Install Python Dependencies from the updated requirements.txt
# This will now install bitsandbytes and accelerate
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (punkt) during the build
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt
# Set environment variable so NLTK knows where to find data
ENV NLTK_DATA=/usr/local/share/nltk_data

# Copy the rest of your application code into the container
# Ensure static and templates directories are copied
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Command to run the application using Gunicorn when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]