# DeepTE-Peptide — Reproducible container image
# Multi-modal deep learning for pan-cancer TE-derived peptide immunogenicity prediction

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="DeepTE-Peptide Research Team"
LABEL description="DeepTE-Peptide: pan-cancer TE-peptide immunogenicity prediction"
LABEL version="0.1.0"

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        tar \
        bzip2 \
        ca-certificates \
        libssl-dev \
        libcurl4-openssl-dev \
        libxml2-dev \
        libbz2-dev \
        liblzma-dev \
        zlib1g-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libhts-dev \
        tabix \
        bgzip \
        samtools \
        bedtools \
        hmmer \
        ncbi-blast+ \
        vim \
        less \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Miniconda
# ---------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda clean -a -y

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
WORKDIR /app
COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml -n deepte && conda clean -a -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "deepte", "/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY . /app
RUN pip install -e . --no-deps

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV DEEPTE_DATA_DIR=/app/data
ENV DEEPTE_RESULTS_DIR=/app/results
ENV DEEPTE_CONFIG_DIR=/app/config
ENV DETERMINISTIC_CUDNN=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
VOLUME ["/app/data", "/app/results"]

# ---------------------------------------------------------------------------
# Default command
# ---------------------------------------------------------------------------
WORKDIR /app
ENTRYPOINT ["conda", "run", "-n", "deepte"]
CMD ["bash", "scripts/run_pipeline.sh", "--help"]
