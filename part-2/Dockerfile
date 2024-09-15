# Use the official Python image with a specific version
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the local project files to the working directory
COPY . .

# Install system dependencies (for selenium and ChromeDriver)
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    chromium-driver \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port where the Streamlit app will run
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
