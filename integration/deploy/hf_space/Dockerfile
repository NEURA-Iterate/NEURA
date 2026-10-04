# Hugging Face Docker Space for the NEURA classifier. Built from a staged copy of the repo
# (see deploy.py), so paths are relative to the repo root.
FROM node:20-slim AS web
WORKDIR /app/integration/frontend
COPY integration/frontend/package.json integration/frontend/package-lock.json ./
RUN npm ci
COPY integration/frontend ./
RUN npm run build

FROM python:3.10-slim
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    NEURA_DATA_DIR=/home/user/data/neura \
    NEURA_DEMO_SAMPLES=0 \
    NEURA_JOBS_DIR=/home/user/data/runs/app_jobs \
    PORT=7860
RUN pip install --no-cache-dir torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu124
WORKDIR /app
COPY --chown=user . /app
COPY --chown=user --from=web /app/integration/frontend/dist /app/integration/frontend/dist
RUN pip install --no-cache-dir -e /app/integration/backend \
    && mkdir -p /home/user/data/runs /home/user/.cache && chown -R user:user /home/user
USER user
EXPOSE 7860
CMD ["bash", "/app/integration/deploy/hf_space/start.sh"]
