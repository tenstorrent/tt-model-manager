# syntax=docker/dockerfile:1.7-labs
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# The tt-model image: a self-contained serving container for ONE model.
#
# Built by `tt-model package <manifest.yaml>` — build.py stages a small context
# (this file, code/, install_engine.sh, verify.sh, entrypoint.sh, requirements.lock)
# and passes the tt-metal tree as the named build context `metalsrc` (a local checkout
# path, hermetic, the default — or an empty dir when METAL_MODE=git).
#
# Type-specific work (vLLM vs the vllm fork, verification assertions) arrives through
# two generated scripts, install_engine.sh and verify.sh, so this file never changes
# per model type. Plain `docker build` with BuildKit; NOT buildx bake.
#
# Everything unusual here is a bug someone already paid for — see docs/packaging.md.

ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
ARG METAL_MODE=local
ARG UBUNTU_VERSION=22.04

#############################################################
# prep: the venv, BEFORE build_metal.sh — it bakes `command -v python3` into
# -DPython3_EXECUTABLE, so our interpreter must already own PATH (the dev image
# ships its own venv at /opt/venv, which must not win).
#############################################################
FROM ${BASE_IMAGE} AS prep

ARG PYTHON_VERSION=3.12
ENV UV_HTTP_RETRIES=10
# keep the vllm sdist + wheels built from it out of the image layers
ENV UV_NO_CACHE=1
ENV UV_PYTHON_INSTALL_DIR=/usr/local/share/uv
ENV VENV=/opt/tt-venv

RUN uv venv --python "${PYTHON_VERSION}" --relocatable --link-mode copy "$VENV" \
    # `wheel` is required: tt-metal setup.py's EditableWheel opens the built wheel to
    # inject ttnn-custom.pth; without it the editable install silently loses the .pth.
    && uv pip install --python "$VENV/bin/python" \
       setuptools==80.10.2 setuptools-scm==8.1.0 wheel
ENV VIRTUAL_ENV=$VENV
ENV PATH=$VENV/bin:$PATH

#############################################################
# metal source: local build context (hermetic — packages the tree that was
# validated, uncommitted work included) or a git clone (CI; reproducible from a sha).
#############################################################
FROM prep AS metal-local
# models/ is excluded ON PURPOSE: the staged code/ (the manifest's allowlist) becomes
# the ONLY `models` package in the image, so an under-specified allowlist fails the
# verify step here, on the author's machine — not on a consumer's pull.
COPY --from=metalsrc \
    --exclude=.git --exclude=.cpmcache --exclude=build_Release* --exclude=build \
    --exclude=built --exclude=python_env --exclude=.venv* --exclude=generated \
    --exclude=models --exclude=docs --exclude=tech_reports --exclude=tests \
    --exclude=tt-train --exclude=model_tracer --exclude=.github \
    . /opt/tt-metal

FROM prep AS metal-git
ARG METAL_GIT_REPO
ARG METAL_GIT_REF
RUN git clone --filter=blob:none "${METAL_GIT_REPO}" /opt/tt-metal \
    && git -C /opt/tt-metal checkout "${METAL_GIT_REF}" \
    && git -C /opt/tt-metal submodule update --init --recursive --depth 1 \
    && rm -rf /opt/tt-metal/.git /opt/tt-metal/models /opt/tt-metal/docs \
       /opt/tt-metal/tech_reports /opt/tt-metal/tests /opt/tt-metal/tt-train

#############################################################
# builder
#############################################################
FROM metal-${METAL_MODE} AS builder

ARG SCM_VERSION
WORKDIR /opt/tt-metal

# The model's own code — needed in the builder too: the engine step may pip-install
# the model's vllm extension out of it.
COPY code/ /opt/tt-metal/

# The long one (~1.5-2.5 h cold on 16 cores). The cache mounts are the highest-leverage
# lines in this file: ccache + the CPM source cache (~3.6 GB incl. emsdk) live outside
# the image, so iterating on any LATER stage does not re-pay the C++ build.
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/cpm \
    ./build_metal.sh --enable-ccache --cpm-source-cache /cpm

# Editable install of ttnn from this tree (setup.py skips cmake for srcdir installs —
# the build above produced _ttnn.so; this step wires the tree up and writes
# ttnn-custom.pth). Two quirks, both from tt-metal's setup.py:
#   - it asserts <root>/.git EXISTS for a srcdir editable install; an empty dir passes.
#   - SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TTNN skips SCM entirely, so no git metadata
#     (and no reference to the fork) is ever inside the image.
RUN mkdir -p /opt/tt-metal/.git \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TTNN="${SCM_VERSION}" \
       uv pip install --python "$VENV/bin/python" --no-build-isolation -e /opt/tt-metal

# The serving stack — generated per model type by build.py (see tt_model/types/).
# tt-smi rides along so `tt-model stop` can reset the mesh from a throwaway container.
COPY install_engine.sh requirements.lock* /ctx/
RUN bash /ctx/install_engine.sh \
    && uv pip install --python "$VENV/bin/python" tt-smi

# vllm-legacy installs the fork editable at /opt/vllm and it must survive into the
# runtime stage; for type vllm this stays an empty dir so the COPY below never fails.
# Same trick for /opt/tenstorrent (system sfpi) when the base image lacks it — the JIT
# uses /opt/tt-metal/runtime/sfpi first, which we always ship.
RUN mkdir -p /opt/vllm /opt/tenstorrent

# Prune: keep exactly what serving loads. build_Release keeps only lib/ (the RUNPATH
# target); runtime/ keeps sfpi (the JIT's RISC-V cross-compiler — consulted before
# /opt/tenstorrent/sfpi, and required at serve time) and hw/ (crt0/noc .o + .ld).
RUN rm -rf /opt/tt-metal/.cpmcache /opt/tt-metal/.git /opt/tt-metal/infra \
       /opt/tt-metal/jobs /opt/tt-metal/releases /opt/tt-metal/dockerfile \
       /opt/tt-metal/contributing \
    && find /opt/tt-metal/build_Release -mindepth 1 -maxdepth 1 ! -name lib -exec rm -rf {} + \
    && ln -sfn build_Release /opt/tt-metal/build \
    # bytecode for the read-only trees: the runtime user cannot write .pyc next to the
    # sources, so every boot would silently re-parse them otherwise
    && "$VENV/bin/python" -m compileall -j0 -q \
       /opt/tt-metal/ttnn /opt/tt-metal/models /opt/tt-metal/tools || true

#############################################################
# runtime
#############################################################
FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG OMPI_DIR=/opt/openmpi-v5.0.7-ulfm
ARG EXTRA_MODELS_DIR=""
ARG TT_MODEL_TYPE
ARG MODEL_NAME
ARG MODEL_REPO
ARG MODEL_WEIGHTS
ARG MODEL_ARCH
ARG MODEL_PROFILES

# The apt set is the built artifacts' actual link closure (readelf -d), nothing more.
# libmpi comes from the ULFM OpenMPI COPY below — Ubuntu's libopenmpi3 ships
# libopen-pal.so.40, not the .so.80 these binaries need; apt cannot satisfy it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhwloc15 libnuma1 libatomic1 libudev1 libcap2 zlib1g \
    libevent-core-2.1-7 libevent-pthreads-2.1-7 \
    libgl1 libsndfile1 ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# A REAL passwd entry at uid 1000: tt-metal derives its JIT cache dir from $HOME, and
# a --user with no passwd entry gets HOME=/ (unwritable → the JIT build dies). Host
# uid 1000 also matches bind-mount ownership. (The dev image's own user is 1001.)
RUN useradd --uid 1000 --create-home --home-dir /home/tt --shell /bin/bash tt \
    && mkdir -p /home/tt/work/logs /cache && chown -R tt:tt /home/tt /cache

# Stable -> volatile, so a model edit re-uploads a 5 MB layer, not a 3 GB one.
COPY --from=builder ${OMPI_DIR} ${OMPI_DIR}
COPY --from=builder /opt/tenstorrent /opt/tenstorrent
COPY --from=builder /usr/local/share/uv /usr/local/share/uv
COPY --from=builder /opt/tt-venv /opt/tt-venv
COPY --from=builder /opt/vllm /opt/vllm
COPY --from=builder /opt/tt-metal/runtime /opt/tt-metal/runtime
COPY --from=builder /opt/tt-metal/build_Release /opt/tt-metal/build_Release
COPY --from=builder /opt/tt-metal/build /opt/tt-metal/build
COPY --from=builder /opt/tt-metal/tt_metal /opt/tt-metal/tt_metal
COPY --from=builder /opt/tt-metal/ttnn /opt/tt-metal/ttnn
COPY --from=builder /opt/tt-metal/tools /opt/tt-metal/tools
COPY --from=builder /opt/tt-metal/setup.py /opt/tt-metal/pyproject.toml /opt/tt-metal/
COPY code/ /opt/tt-metal/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

ENV VENV=/opt/tt-venv
ENV VIRTUAL_ENV=/opt/tt-venv
ENV PATH=/opt/tt-venv/bin:$PATH
# TT_METAL_RUNTIME_ROOT is the live variable; it also beats the compiled-in
# /usr/libexec/tt-metalium probe that would otherwise win over ttnn's set_root_dir.
# TT_METAL_HOME is set for anything older that still reads it.
ENV TT_METAL_RUNTIME_ROOT=/opt/tt-metal
ENV TT_METAL_HOME=/opt/tt-metal
ENV PYTHONPATH=/opt/tt-metal
# BOTH paths: metal's own libs AND the ULFM MPI. Overwriting one with the other is a
# known upstream bug pattern — do not "simplify" this line.
ENV LD_LIBRARY_PATH=/opt/tt-metal/build/lib:${OMPI_DIR}/lib
ENV EXTRA_MODELS_DIR=${EXTRA_MODELS_DIR}
ENV TT_VLLM_BUILTIN_MODELS=0
ENV TT_MODEL_TYPE=${TT_MODEL_TYPE}
ENV HF_HOME=/hf
ENV TT_METAL_CACHE=/cache

USER tt
WORKDIR /home/tt/work

# Build-time verification — what makes the prune and the code/ allowlist safe. The
# assertions are generated per model (tt_model/types/*.verify_lines); an under-shipped
# image fails HERE, not ten minutes into a consumer's first boot.
COPY verify.sh /ctx/verify.sh
RUN bash /ctx/verify.sh

LABEL org.tenstorrent.tt-model="${MODEL_NAME}" \
      org.tenstorrent.tt-model.repo="${MODEL_REPO}" \
      org.tenstorrent.tt-model.weights="${MODEL_WEIGHTS}" \
      org.tenstorrent.tt-model.arch="${MODEL_ARCH}" \
      org.tenstorrent.tt-model.type="${TT_MODEL_TYPE}" \
      org.tenstorrent.tt-model.profiles="${MODEL_PROFILES}"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
