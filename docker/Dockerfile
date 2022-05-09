FROM python:3.6.15-slim

RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gifsicle \
    gcc \
    libgmp3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV APPPATH /home/nn_robustness_analysis
COPY . $APPPATH
WORKDIR $APPPATH

RUN python -m pip install --no-cache-dir torch==1.10.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install --no-cache-dir tensorflow-cpu
RUN python -m pip install --no-cache-dir -e crown_ibp 
RUN python -m pip install --no-cache-dir -e auto_LiRPA
RUN python -m pip install --no-cache-dir -e robust_sdp
RUN python -m pip install --no-cache-dir -e nn_partition
RUN python -m pip install --no-cache-dir -e nn_closed_loop
# RUN python -m pip install -e crown_ibp auto_LiRPA robust_sdp nn_partition nn_closed_loop

CMD ["/bin/bash"]