# Use the official lightweight Python 3.10 image
FROM python:3.10-slim

# Create a non-root user (Hugging Face requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY --chown=user . .

# Expose the Hugging Face port
EXPOSE 7860

# Start the FastAPI app on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]