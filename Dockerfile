# Step 3.1: Specify the Base Image (Using Python 3.12)
# We start with an official Python image. '-slim' is smaller than the full version.
FROM python:3.12-slim

# Step 3.2: Set the Working Directory inside the Container
# Subsequent commands like COPY and RUN will happen relative to this path.
WORKDIR /app

# Step 3.3: Install System Dependencies
# Update package lists, install Tesseract, common graphics/utility libraries
# (often needed by Python image/ML libraries), then clean up to keep the image small.
# This is done in one RUN layer for efficiency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Add any other necessary system packages here if you discover errors later
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Step 3.4: Copy ONLY the requirements file first
# This leverages Docker's layer caching. If requirements.txt doesn't change,
# Docker can reuse the cached layer from the next step, speeding up rebuilds.
COPY requirements.txt .

# Step 3.5: Install Python Dependencies
# Upgrade pip first, then install all packages from requirements.txt.
# --no-cache-dir reduces the final image size.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 3.6: Download NLTK 'punkt' data during the build
# Specifies a standard location within the container for NLTK data.
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt

# Step 3.7: Set Environment Variable for NLTK
# Tells the NLTK library inside the container where to find the data we just downloaded.
ENV NLTK_DATA=/usr/local/share/nltk_data

# Step 3.8: Copy the rest of your application code
# This copies app.py, main.py, static/, templates/, etc., into /app inside the container.
COPY . .

# Step 3.9: Expose the port Gunicorn will listen on (Informational)
# This tells Docker the container intends to listen on port 8501. It doesn't publish it yet.
EXPOSE 8501

# Step 3.10: Define the command to run when the container starts
# Uses Gunicorn (a production WSGI server) to run your Flask app (app:app -> app variable in app.py).
# --bind 0.0.0.0:8501: Listens on all network interfaces inside the container on port 8501.
# --workers 1: Crucial for memory-intensive apps, prevents multiple large model copies.
# --threads 4: Allows handling multiple concurrent I/O requests within the worker.
# --timeout 300: Sets a 5-minute timeout for workers, important for long summarization tasks.
CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
