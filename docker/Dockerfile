FROM python:3.10.3-slim

RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gifsicle \
    gcc \
    libgmp3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV APPPATH /home/nn_robustness_analysis
COPY . $APPPATH
WORKDIR $APPPATH

RUN python -m pip install --no-cache-dir torch==1.12.0+cpu torchvision==0.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install --no-cache-dir third_party/crown_ibp third_party/jax_verify third_party/auto_LiRPA .
# RUN python -m pip install --no-cache-dir "jax_verify @ git+https://gitlab.com/mit-acl/ford_ugvs/jax_verify.git" "crown_ibp @ git+https://gitlab.com/mit-acl/ford_ugvs/crown_ibp.git" .

ENV JAX_PLATFORMS cpu
ENV TF_CPP_MIN_LOG_LEVEL '2'

CMD ["/bin/bash"]