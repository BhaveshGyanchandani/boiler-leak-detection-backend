FROM python:3.12-slim

WORKDIR /app

# Copy and install dependencies first (cache optimisation)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY backend/ ./backend/

# Use the PORT environment variable or default to 10000
ENV PORT=10000

EXPOSE $PORT

CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 5
