FROM python:3.6.15

RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gifsicle \
    gcc

ENV APPPATH /home/nn_robustness_analysis
COPY . $APPPATH
WORKDIR $APPPATH

RUN python -m pip install ./crown_ibp ./auto_LiRPA ./robust_sdp ./nn_partition ./nn_closed_loop

CMD ["/bin/bash"]