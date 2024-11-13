FROM ubuntu:22.04
COPY . /cs149_asst4GPT/

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
    
RUN sed "1i nameserver 8.8.8.8" /etc/resolv.conf

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    gcc \
    gdb \
    g++ \
    python3-pip \
    make \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /cs149_asst4GPT
RUN apt-get clean
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.23.5
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.1.2
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tiktoken
RUN apt-get update
RUN apt-get install ninja-build -y
