# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""tt-model: package Tenstorrent models as Docker images and publish them on Hugging Face.

Three jobs, nothing else:

1. ``package`` — build a self-contained Docker image from a working tt-metal fork,
   driven by one YAML manifest.
2. ``push`` — upload the image (as an exploded OCI layout) plus the manifest and the
   model's own code to a Hugging Face model repo.
3. ``pull`` / ``serve`` — download from HF, ``docker load``, and run. Weights live in
   the host's HF cache; everything else lives inside the container.
"""

__version__ = "0.2.0"

# HF model-repo tag that marks a repo as a tt-model package (used by search/filtering).
TT_MODEL_TAG = "tt-model"

# Filename of the manifest at the root of every published repo.
MANIFEST_NAME = "tt-model.yaml"

__all__ = ["TT_MODEL_TAG", "MANIFEST_NAME", "__version__"]
