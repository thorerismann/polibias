# Stage 1: Pull all Ollama models into a cache layer
FROM ollama/ollama:latest AS models

# Pull each model (cached as a Docker layer — only re-downloads when models change)
RUN ollama serve & sleep 3 && \
    ollama pull llama3.2:latest && \
    ollama pull gemma2:latest && \
    ollama pull phi3:mini && \
    ollama pull qwen2.5:3b-instruct && \
    ollama pull gemma3:4b && \
    kill %1 || true

# Stage 2: Runtime image with Ollama + Python + polibias
FROM ollama/ollama:latest

# Install Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-pulled models from stage 1
COPY --from=models /root/.ollama /root/.ollama

# Set up the application
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY data/input_files/ data/input_files/
COPY cloud/config.toml cloud/config.toml
COPY cloud/entrypoint.sh entrypoint.sh

# Install polibias with cloud dependencies
RUN pip3 install --no-cache-dir --break-system-packages ".[cloud]"

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
