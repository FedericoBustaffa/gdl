# Base image: Debian slim with Python 3.12
FROM python:3.12-slim

# Install graphviz (the only system-level dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Mount the notebooks and _static directories
VOLUME /app/notebooks
VOLUME /app/_static

# Expose the default Jupyter Notebook port
EXPOSE 8888

# Command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
