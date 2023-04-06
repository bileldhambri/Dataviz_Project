# Use the Python 3.8 slim-buster base image
FROM python:3.8-slim-buster
# Install dependencies
RUN pip install panel pandas numpy scikit-learn tensorflow
# Copy your application code to the Docker image
COPY . /app
# Set the working directory
WORKDIR /app
# Expose port 8080 for Panel
EXPOSE 8080
# Run your Panel application
CMD ["panel", "serve", "myapp.ipynb", "--port", "8080", "--allow-websocket-origin=*"]
