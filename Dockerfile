# Dockerfile
# Use an official Python runtime based on Debian Bookworm
FROM python:3.11-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    # Add any other system dependencies required by unstructured[all-docs]
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements files into the container at /app
COPY requirements.txt requirements.lock.txt ./

# Install UV package manager and use it to install dependencies from the lock file
# This ensures exact version matching for reproducible builds
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache-dir -r requirements.lock.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Convert start.sh line endings to Unix format using sed and make it executable
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Make port 8000 available for FastAPI and 8501 for Streamlit
EXPOSE 8000 8501

# Define environment variable (optional, can be set at runtime)
# ENV OPENAI_API_KEY=your_api_key_here

# Run the startup script when the container launches
CMD ["./start.sh"]