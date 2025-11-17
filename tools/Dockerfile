FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY backend backend
COPY frontend frontend
COPY ["9 Facial Expressions you need", "9 Facial Expressions you need"]
COPY README.md README.md

ENV PYTHONPATH=/workspace
EXPOSE 8000

CMD ["bash"]
