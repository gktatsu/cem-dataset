# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 必要な Python パッケージをインストール
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy scipy pillow tqdm

# ソースコードをコピー
COPY . /app

ENTRYPOINT ["python", "fid/compute_cem_fid.py"]
# ENTRYPOINT ["python", "fid/compute_normal_fid.py"]
CMD ["--help"]
