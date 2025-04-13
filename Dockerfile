# Use an official Python runtime as a parent image
# Sticking with 3.12, ensure PyTorch build compatibility in requirements.txt
FROM python:3.12-slim

# Set environment variables for cleaner builds and logs
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    # Set NLTK Data path globally
    NLTK_DATA=/usr/local/share/nltk_data

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# Added build-essential, pkg-config often needed for building Python packages
# Combined update, install, and clean in one RUN layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Tesseract OCR Engine + English language pack
    tesseract-ocr \
    tesseract-ocr-eng \
    # Build tools
    build-essential \
    pkg-config \
    # Libraries potentially needed by Pillow, PyMuPDF, etc.
    libc6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Add any other specific system deps your project needs
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy just the requirements file first to leverage Docker cache
COPY requirements.txt .

# --- Hugging Face Token Handling (Choose Option B from previous response if desired) ---
# ARG HF_TOKEN
# ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
# -------------------------------------------------------------------------------------

# Upgrade pip and install Python dependencies
# Ensure requirements.txt has the CORRECT PyTorch version for your HOST's CUDA Driver!
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (punkt) during the build
# The ENV NLTK_DATA above tells NLTK where to look
RUN python -m nltk.downloader -d $NLTK_DATA punkt

# Copy the rest of the application code into the working directory
# Make sure you have a .dockerignore file in your project root!
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run the application using Gunicorn
# REMOVED --preload, kept -w 1 because 'spawn' was set in app.py
CMD ["gunicorn", "-w", "1", "--timeout", "300", "-b", "0.0.0.0:8501", "app:app"]