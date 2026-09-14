FROM python:3.11-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt pyproject.toml README.md ./

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Install the project itself
RUN pip install --no-cache-dir -e .

# Default command: run the safe live simulation (mock adapters)
CMD ["qte-live"]
