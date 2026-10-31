FROM python:3.11-slim-bookworm
ARG VCS_REF=development
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=10000 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VCS_REF=${VCS_REF}
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY deployment/docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY training/drawai/ /app/training/drawai/
COPY deployment/docker/model-bundle/ /app/models/exported/paint35-v1/
COPY LICENSE NOTICE.md /app/
RUN python -c "from training.drawai.paint_package import verify_bundle; verify_bundle('models/exported/paint35-v1','86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2')" && useradd --uid 10001 --create-home appuser
USER 10001
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','10000')+'/health',timeout=4)"
LABEL org.opencontainers.image.title="DrawAI DINOv3 inference API" org.opencontainers.image.source="https://github.com/namphwong/DUST" org.opencontainers.image.revision=${VCS_REF}
CMD ["sh", "-c", "exec uvicorn training.drawai.public_api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --no-access-log --timeout-keep-alive 5"]
