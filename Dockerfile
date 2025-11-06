# Base image
FROM python:3.9-slim

# Prevents Python from writing .pyc files and ensures output is flushed
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PAFY_BACKEND=internal

# System deps needed by Pillow, build, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps first (leverage layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz \
    && python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"


# Copy application code and assets
COPY "app.py" ./
COPY logo2.png ./logo2.png
COPY "Courses.py" ./
RUN mkdir -p Uploaded_Resumes && chmod -R 777 Uploaded_Resumes

# Streamlit config: run headless on provided PORT
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Expose default Streamlit port (App Runner will pass PORT env)
EXPOSE 8501

# Health check (optional but useful locally)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Start the app. App Runner sets PORT; default to 8501 for local runs.
CMD ["/bin/sh", "-c", "streamlit run 'app.py' --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]


