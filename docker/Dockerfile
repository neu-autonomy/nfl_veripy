FROM python:3.6.15-slim

RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gifsicle

ENV APPPATH /home/myapp
COPY . $APPPATH
WORKDIR $APPPATH

RUN ls
RUN ls /home
RUN ls /home/myapp

RUN python -m pip install ./crown_ibp ./auto_LiRPA ./robust_sdp ./nn_partition ./nn_closed_loop

CMD ["/bin/bash"]