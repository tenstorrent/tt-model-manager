# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""tt-model: publish and pull self-contained tt-metal model bundles over Hugging Face Hub."""

__version__ = "0.1.0"

# HF model-repo tag that marks a repo as a tt-model bundle (used by search).
TT_MODEL_TAG = "tt-model-cache"

# HF model-repo tag that OPTS a (public) bundle into the community catalog. This is a
# deliberate, separate act from `push`: pushing a bundle only tags it TT_MODEL_TAG;
# `--publish` (or `tt-model publish`) additionally adds this tag. The web catalog lists
# ONLY repos carrying this tag. tt-model stores nothing — the catalog is a pure index of
# pointers to public HF repos, which remain under their owners' governance.
TT_MODEL_CATALOG_TAG = "tt-model-catalog"

# Filename of the compatibility manifest at the root of every bundle.
MANIFEST_NAME = "tt_kernel_manifest.json"

__all__ = [
    "TT_MODEL_TAG",
    "TT_MODEL_CATALOG_TAG",
    "MANIFEST_NAME",
]
