FROM ubuntu@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils \
        g++ \
        libc-bin \
        libopencv-imgcodecs-dev=4.6.0+dfsg-13.1ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
