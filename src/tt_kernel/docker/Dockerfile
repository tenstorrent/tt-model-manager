# syntax=docker/dockerfile:1.7-labs
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# The tt-model container image: a self-contained serving container for ONE model.
#
# Built by `tt-model package --container <manifest.yaml>`, which stages a small context
# (this file, code/, install_engine.sh, verify.sh, entrypoint.sh, requirements.lock) and
# passes the tt-metal tree as the named build context `metalsrc` — a local checkout
# (hermetic, the default) or an empty dir when METAL_MODE=git.
#
# Kind-specific work (installing the serving stack, the verification assertions) arrives
# through two GENERATED scripts, install_engine.sh and verify.sh, so this file never
# changes per kind. Plain `docker build` with BuildKit; NOT buildx bake.
#
# Everything unusual here is a bug someone already paid for. Do not "simplify" a line
# without reading its comment.

ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
ARG METAL_MODE=local
ARG UBUNTU_VERSION=22.04

#############################################################
# prep: the venv, BEFORE build_metal.sh — it bakes `command -v python3` into
# -DPython3_EXECUTABLE, so our interpreter must already own PATH (the dev image ships
# its own venv at /opt/venv, which must not win).
#############################################################
FROM ${BASE_IMAGE} AS prep

ARG PYTHON_VERSION=3.12
ENV UV_HTTP_RETRIES=10
# keep the vllm sdist and the wheels built from it out of the image layers
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
# metal source: a local build context (hermetic — packages the tree that was validated,
# uncommitted work included) or a git clone (CI; reproducible from a sha).
#############################################################
FROM prep AS metal-local
# models/ is excluded ON PURPOSE: the staged code/ (the manifest's allowlist) becomes the
# ONLY `models` package in the image, so an under-specified allowlist fails the verify
# step below, on the author's machine — not on a consumer's first boot.
COPY --from=metalsrc \
    --exclude=.git --exclude=.cpmcache --exclude=build_Release* --exclude=build \
    --exclude=built --exclude=python_env --exclude=.venv* --exclude=generated \
    --exclude=models --exclude=docs --exclude=tech_reports \
    --exclude=tt-train --exclude=model_tracer --exclude=.github \
    . /opt/tt-metal

FROM prep AS metal-git
ARG METAL_GIT_REPO
ARG METAL_GIT_REF
RUN git clone --filter=blob:none "${METAL_GIT_REPO}" /opt/tt-metal \
    && git -C /opt/tt-metal checkout "${METAL_GIT_REF}" \
    && git -C /opt/tt-metal submodule update --init --recursive --depth 1 \
    && rm -rf /opt/tt-metal/.git /opt/tt-metal/models /opt/tt-metal/docs \
       /opt/tt-metal/tech_reports /opt/tt-metal/tt-train

#############################################################
# builder
#############################################################
FROM metal-${METAL_MODE} AS builder

ARG SCM_VERSION
ARG METAL_DESCRIBE=v0.0.0
WORKDIR /opt/tt-metal

# The tree ships without .git, but tt-metal's CMake derives PROJECT_VERSION from
# `git describe` (cmake/version.cmake ParseGitDescribe) and configuration FAILS on an
# empty version. A one-empty-commit repo tagged with the HOST checkout's describe string
# reproduces the real version — and satisfies setup.py's srcdir .git assertion — without
# any actual history entering the image (it is pruned before the runtime stage).
RUN git init -q /opt/tt-metal \
    && git -C /opt/tt-metal -c user.email=tt-model@tenstorrent.com -c user.name=tt-model \
       commit -q --allow-empty -m "tt-model version stub" \
    && git -C /opt/tt-metal tag "${METAL_DESCRIBE}"

# The model's own code — needed in the builder too: the engine step may pip-install the
# model's vLLM extension out of it.
COPY code/ /opt/tt-metal/

# The long one (~1.5-2.5 h cold on 16 cores). The cache mounts are the highest-leverage
# lines in this file: ccache and the CPM source cache (~3.6 GB incl. emsdk) live OUTSIDE
# the image, so iterating on any later stage does not re-pay the C++ build.
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/cpm \
    ./build_metal.sh --enable-ccache --cpm-source-cache /cpm

# Editable install of ttnn from this tree (setup.py skips cmake for srcdir installs — the
# build above produced _ttnn.so; this step wires the tree up and writes ttnn-custom.pth).
# SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TTNN makes scm take the version as-is: no SCM lookup,
# so no fork metadata is ever consulted or embedded.
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TTNN="${SCM_VERSION}" \
    uv pip install --python "$VENV/bin/python" --no-build-isolation -e /opt/tt-metal

# The serving stack — generated per kind (see tt_kernel/launchers.py). tt-smi rides along
# so `tt-model stop` can reset the mesh from a throwaway container without a host tt-smi.
# plugin-src is the author's local plugin checkout when runtime.plugin.path is set, and
# an empty directory otherwise — `package` always creates it so this COPY cannot fail.
COPY install_engine.sh requirements.lock* /ctx/
COPY plugin-src /ctx/plugin-src
# A local vLLM source tree and any local wheels (the author's own vLLM build, plus
# extras). Both are empty directories unless the manifest asks for them.
COPY vllm-src /ctx/vllm-src
COPY wheels /ctx/wheels
RUN bash /ctx/install_engine.sh \
    && uv pip install --python "$VENV/bin/python" tt-smi

# The fork installs editable at /opt/vllm and must survive into the runtime stage. The
# mkdir keeps the unconditional COPY below valid for any future kind that has no fork.
# Same trick for /opt/tenstorrent (system sfpi) when the base image lacks it — the JIT
# consults /opt/tt-metal/runtime/sfpi first, which we always ship.
RUN mkdir -p /opt/vllm /opt/tenstorrent

# TODO(size): measured against a real tt-metal tree, the runtime stage still carries
# three large items worth revisiting ONCE A BUILD IS GREEN — never before, because
# over-pruning does not fail here: verify.sh catches import-time breakage, but a missing
# JIT input fails ~10 minutes into a CONSUMER's first boot.
#   tools/triage        ~75M  — a debugging tool; no role in serving. The clear win.
#   tt_metal/pre-compiled ~187M — hash-named artifacts; determine whether these duplicate
#                                 the mounted /cache before touching them.
#   runtime/sfpi        ~436M — the JIT's RISC-V cross-compiler. The biggest single item
#                               and NOT removable: serve-time kernel compiles need it.
# Sequence: green build -> measure real layer sizes -> drop triage -> investigate
# pre-compiled -> re-verify with a FULL BOOT, not just verify.sh.

# Prune: keep exactly what serving loads. build_Release keeps only lib/ (the RUNPATH
# target); runtime/ keeps sfpi (the JIT's RISC-V cross-compiler — consulted before
# /opt/tenstorrent/sfpi, and required at serve time) and hw/ (crt0/noc .o and .ld).
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
ARG TT_MODEL_KIND
ARG MODEL_NAME
ARG MODEL_REPO
ARG MODEL_WEIGHTS
ARG MODEL_ARCH
ARG MODEL_PROFILES

# The apt set is the built artifacts' actual link closure (readelf -d), nothing more.
# libmpi comes from the ULFM OpenMPI COPY below — Ubuntu's libopenmpi3 ships
# libopen-pal.so.40, not the .so.80 these binaries need; apt cannot satisfy it.
# libmpc3/libmpfr6/libgmp10/libzstd1: the JIT's RISC-V cross-compiler (sfpi cc1plus,
# lto1) is a HOST binary that links them — without them the first kernel compile dies
# with "cc1plus: error while loading shared libraries: libmpc.so.3".
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhwloc15 libnuma1 libatomic1 libudev1 libcap2 zlib1g \
    libmpc3 libmpfr6 libgmp10 libzstd1 \
    libevent-core-2.1-7 libevent-pthreads-2.1-7 \
    libgl1 libsndfile1 ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# A real user for the BUILD-time verification, plus dirs any uid can write at RUN time.
#
# tt-metal derives its JIT cache dir from $HOME, so HOME must exist and be writable. The
# obvious approach — one baked user whose uid matches the host's — does not survive
# contact with reality: `serve` passes `--user <host uid>:<host gid>` so that files
# written into the bind-mounted HF cache and kernel cache belong to the person running
# it, and a consumer's uid is whatever it is (1000 and 1001 are both common). A uid with
# no passwd entry gets HOME=/ from glibc, which is unwritable, which kills the JIT.
#
# So: keep a named user for the build (USER tt below, when no bind mounts exist), set
# HOME explicitly in ENV so it does not depend on a passwd lookup, and make the runtime
# dirs world-writable with the sticky bit — the same trick /tmp uses — so an arbitrary
# uid can write them without the image knowing which uid that will be.
#
# Ubuntu 24.04's base image already ships a user at uid 1000 (`ubuntu`); 22.04 did not.
# So free the id first, by whatever name currently holds it, instead of assuming it is
# vacant — otherwise `useradd` fails with "UID 1000 is not unique".
RUN existing="$(getent passwd 1000 | cut -d: -f1)" \
    && if [ -n "$existing" ]; then \
         userdel -r "$existing" 2>/dev/null || userdel "$existing"; \
       fi \
    && useradd --uid 1000 --create-home --home-dir /home/tt --shell /bin/bash tt \
    && mkdir -p /home/tt/work/logs /cache /opt/tt-metal \
    && chown -R tt:tt /home/tt /cache /opt/tt-metal \
    && chmod 1777 /home/tt /home/tt/work /home/tt/work/logs /cache

# Stable -> volatile, so a model edit re-uploads a 5 MB layer, not a 3 GB one.
COPY --from=builder ${OMPI_DIR} ${OMPI_DIR}
COPY --from=builder /opt/tenstorrent /opt/tenstorrent
COPY --from=builder /usr/local/share/uv /usr/local/share/uv
COPY --from=builder /opt/tt-venv /opt/tt-venv
COPY --from=builder /opt/vllm /opt/vllm
# --chown=tt: python code in the tree writes into TT_METAL_HOME at import/serve time
# (tools/tracy/common.py mkdirs build/profiler/build_wasm/traces on IMPORT; profiler
# artifacts default under generated/). Root-owned trees turn those into EACCES ten
# minutes into a boot, so the model tree belongs to the runtime user.
COPY --from=builder --chown=tt:tt /opt/tt-metal/runtime /opt/tt-metal/runtime
COPY --from=builder --chown=tt:tt /opt/tt-metal/build_Release /opt/tt-metal/build_Release
COPY --from=builder --chown=tt:tt /opt/tt-metal/build /opt/tt-metal/build
COPY --from=builder --chown=tt:tt /opt/tt-metal/tt_metal /opt/tt-metal/tt_metal
COPY --from=builder --chown=tt:tt /opt/tt-metal/ttnn /opt/tt-metal/ttnn
COPY --from=builder --chown=tt:tt /opt/tt-metal/tools /opt/tt-metal/tools
COPY --from=builder --chown=tt:tt /opt/tt-metal/setup.py /opt/tt-metal/pyproject.toml /opt/tt-metal/
COPY --chown=tt:tt code/ /opt/tt-metal/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# The default profile's env + launch command, generated per model. Gives the image a
# CMD, so `docker run <image>` with the right host flags serves correctly instead of
# starting a misconfigured server — and refuses with the required flags when it cannot.
COPY --chmod=0755 serve-default.sh /usr/local/bin/serve-default.sh

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
# "0" ONLY when the model ships its own vLLM extension (register solely from
# EXTRA_MODELS_DIR); empty for a model that lives in the plugin's BUILTIN registry —
# setting 0 there would register zero architectures. entrypoint.sh unsets an empty value
# so the plugin sees its own default.
ARG TT_VLLM_BUILTIN_MODELS=""
ENV TT_VLLM_BUILTIN_MODELS=${TT_VLLM_BUILTIN_MODELS}
ENV TT_MODEL_KIND=${TT_MODEL_KIND}
ENV HF_HOME=/hf
ENV TT_METAL_CACHE=/cache
# Explicit, so nothing depends on a passwd lookup for whatever uid `serve` passes.
#
# HOME: tt-metal and ttnn derive cache paths from it, and glibc hands an unknown uid
# HOME=/ (unwritable), which kills the JIT.
#
# USER/LOGNAME: `getpass.getuser()` reads LOGNAME, USER, LNAME, USERNAME in that order
# and only falls back to pwd.getpwuid() if none is set — which raises KeyError for a uid
# with no passwd entry. torch's inductor calls it to name its cache dir, so without this
# the server dies on import with "getpwuid(): uid not found". Setting the name here beats
# making /etc/passwd writable, which is the usual workaround.
ENV HOME=/home/tt
ENV USER=tt
ENV LOGNAME=tt

USER tt
WORKDIR /home/tt/work

# Build-time verification — what makes the prune and the code/ allowlist safe. The
# assertions are generated per model (launchers.verify_lines + the manifest's own
# verify: list); an under-shipped image fails HERE, on the author's machine, not ten
# minutes into a consumer's first boot.
COPY verify.sh /ctx/verify.sh
RUN bash /ctx/verify.sh

# Verification runs as `tt` (uid 1000) and its imports CREATE cache dirs under $HOME —
# matplotlib, ttnn, torch, vllm all do. Those land 0755 owned by 1000, so the arbitrary
# uid `serve` passes cannot write inside them, and vLLM then fails to save its model-info
# cache with EACCES. Re-open the tree AFTER verification, so whatever it created is
# covered too; chmod'ing only the dirs that exist BEFORE this RUN is not enough.
# a+rwX: read/write for all, execute only where it already applies (directories).
USER root
RUN chmod -R a+rwX /home/tt
USER tt

LABEL org.tenstorrent.tt-model="${MODEL_NAME}" \
      org.tenstorrent.tt-model.repo="${MODEL_REPO}" \
      org.tenstorrent.tt-model.weights="${MODEL_WEIGHTS}" \
      org.tenstorrent.tt-model.arch="${MODEL_ARCH}" \
      org.tenstorrent.tt-model.kind="${TT_MODEL_KIND}" \
      org.tenstorrent.tt-model.profiles="${MODEL_PROFILES}"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Overridden by any command given to `docker run`; tt-model always passes one.
CMD ["/usr/local/bin/serve-default.sh"]
