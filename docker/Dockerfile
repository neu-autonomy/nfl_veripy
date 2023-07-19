FROM python:3.10.3-slim

RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gifsicle \
    gcc \
    libgmp3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV APPPATH /home/nfl_veripy
COPY . $APPPATH
WORKDIR $APPPATH

# Install pytorch's cpu-only module (much much smaller -> faster download)
RUN python -m pip install --no-cache-dir \
    torch==1.12.0+cpu \
    torchvision==0.13.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install our local (non-pypi) dependencies and nfl_veripy
RUN python -m pip install --no-cache-dir -e \
    third_party/crown_ibp \
    third_party/jax_verify \
    third_party/auto_LiRPA
# RUN python -m pip install --no-cache-dir "jax_verify @ git+https://gitlab.com/mit-acl/ford_ugvs/jax_verify.git" "crown_ibp @ git+https://gitlab.com/mit-acl/ford_ugvs/crown_ibp.git" .

# separately install our pkg
# (after transferring from mit-acl to neu-autonomy, mfe separated this since keeping it as 1 pip install made committed changes not take effect)
RUN python -m pip install --no-cache-dir -e .

# Suppress annoying warnings from tensorflow & jax
ENV JAX_PLATFORMS cpu
ENV TF_CPP_MIN_LOG_LEVEL '2'

CMD ["/bin/bash"]