FROM ghcr.io/andrewembry312-hub/elabs-server/wan26-t2i-runpod:latest
LABEL maintainer="E-Labs AI Studio" description="Wan 2.6 T2I — Text to Image on RunPod"
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY handler.py /workspace/handler.py
WORKDIR /workspace
CMD ["python", "-u", "handler.py"]
