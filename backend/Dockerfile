FROM python:3.12-slim

RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Step 1: Install all deps (nudenet brings onnxruntime CPU)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY app/ ./app/

# Step 2: Pre-download models while onnxruntime CPU works (no GPU needed)
RUN python -c "from nudenet import NudeDetector; NudeDetector()"
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

# Step 3: Replace onnxruntime CPU with GPU version + install CUDA runtime libs
RUN pip uninstall -y onnxruntime && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    onnxruntime-gpu \
    nvidia-cublas-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-curand-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cudnn-cu12

# Make CUDA libs discoverable by onnxruntime at runtime
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:${LD_LIBRARY_PATH}"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
