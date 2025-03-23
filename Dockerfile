# Use a Linux-based image with Python 3.11
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Expose ports (uncomment later if needed)
# EXPOSE 8888 5000 3000

# Start with Bash (instead of Jupyter) for flexibility
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
