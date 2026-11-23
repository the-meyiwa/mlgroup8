FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /srv

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY features.json ./features.json
COPY api ./api
COPY api/model/model.joblib ./api/model/model.joblib
COPY api/model/model.meta.json ./api/model/model.meta.json

ENV MODEL_PATH=/srv/api/model/model.joblib
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
