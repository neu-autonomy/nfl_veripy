FROM python:3.6.15

RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gifsicle \
    gcc

ENV APPPATH /home/nn_robustness_analysis
COPY . $APPPATH
WORKDIR $APPPATH

RUN python -m pip install -e crown_ibp 
RUN python -m pip install -e auto_LiRPA
RUN python -m pip install -e robust_sdp
RUN python -m pip install -e nn_partition
RUN python -m pip install -e nn_closed_loop
# RUN python -m pip install -e crown_ibp auto_LiRPA robust_sdp nn_partition nn_closed_loop

CMD ["/bin/bash"]