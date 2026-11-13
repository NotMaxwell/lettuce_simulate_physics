# Dev-only image: building, tests, headless bench, and sandboxed Claude Code sessions.
# Not part of the shipped game or its build (see CLAUDE.md "Hard rules").
FROM debian:trixie-slim

ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256=1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026
# Match your host UID so files written to the bind-mounted repo stay yours.
ARG USER_UID=1000
ARG USER_GID=1000

# mesa-vulkan-drivers gives RADV/ANV when /dev/dri is passed in, and lavapipe (CPU Vulkan)
# when it is not, so GPU tests still run (slowly) on machines without a usable GPU.
# libpulse0/libasound2t64 are only dlopened by SDL; with SDL_AUDIODRIVER=dummy they are unused.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils git less procps ripgrep \
        mesa-vulkan-drivers libvulkan1 vulkan-tools \
        libpulse0 libasound2t64 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/zig.tar.xz \
        "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
    && echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - \
    && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
    && ln -s /opt/zig/zig /usr/local/bin/zig && rm /tmp/zig.tar.xz

RUN groupadd -g "${USER_GID}" dev && useradd -m -u "${USER_UID}" -g dev -s /bin/bash dev
USER dev
WORKDIR /home/dev

# Created here so the named volumes mounted over them inherit dev's ownership, not root's.
RUN mkdir -p /home/dev/.cache/zig /home/dev/.claude

RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/dev/.local/bin:${PATH}" \
    CLAUDE_CONFIG_DIR=/home/dev/.claude \
    SDL_VIDEODRIVER=offscreen \
    SDL_AUDIODRIVER=dummy
