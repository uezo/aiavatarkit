FROM debian:bookworm-slim

ARG ASTERISK_VERSION=22.10.1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        libcap-dev \
        libcurl4-openssl-dev \
        libedit-dev \
        libjansson-dev \
        libncurses-dev \
        libnewt-dev \
        libsqlite3-dev \
        libssl-dev \
        liburiparser-dev \
        libxml2-dev \
        patch \
        perl \
        pkg-config \
        python3 \
        uuid-dev \
    && curl --fail --show-error --location \
        "https://downloads.asterisk.org/pub/telephony/asterisk/asterisk-${ASTERISK_VERSION}.tar.gz" \
        --output /tmp/asterisk.tar.gz \
    && mkdir /usr/src/asterisk \
    && tar --extract --gzip --file /tmp/asterisk.tar.gz \
        --directory /usr/src/asterisk --strip-components=1 \
    && cd /usr/src/asterisk \
    && ./configure \
        --with-pjproject-bundled \
        --without-dahdi \
        --without-pri \
        --without-x11 \
    && make menuselect.makeopts \
    && menuselect/menuselect \
        --enable app_echo \
        --enable app_stasis \
        --enable bridge_softmix \
        --enable chan_websocket \
        --enable codec_resample \
        --enable res_ari \
        --enable res_ari_channels \
        --enable res_http_websocket \
        --enable res_websocket_client \
        menuselect.makeopts \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/asterisk.tar.gz /usr/src/asterisk /var/lib/apt/lists/*

COPY config/ /etc/asterisk/

EXPOSE 5060/udp 8088 10000-10020/udp

HEALTHCHECK --interval=2s --timeout=2s --start-period=5s --retries=30 \
    CMD asterisk -rx "core show version" || exit 1

CMD ["asterisk", "-f", "-vvv"]
