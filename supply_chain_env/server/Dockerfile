FROM python:3.11-slim

WORKDIR /app

# Copy requirements based on context from repo root
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Symlink so that absolute python imports from 'supply_chain_env' resolve properly
RUN ln -s /app /app/supply_chain_env
ENV PYTHONPATH="/app"

EXPOSE 8000

# Healthcheck to ensure it returns ok within 10 seconds of startup
HEALTHCHECK --interval=2s --timeout=2s --start-period=1s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
