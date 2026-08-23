# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The enumeration of model types — the single source of truth.

Mirrored, in prose, by docs/model_types.md. Keep the two in step.

| type          | vLLM from                          | TT plugin from                    | launched with                |
| ------------- | ---------------------------------- | --------------------------------- | ---------------------------- |
| ``vllm``        | stock sdist (VLLM_TARGET_DEVICE=empty) | standalone tenstorrent/vllm-tt-plugin | ``vllm serve``                 |
| ``vllm-legacy`` | the tenstorrent/vllm fork, editable    | the fork's in-tree plugins/vllm-tt-plugin | ``run_vllm_server --stages serve`` |
"""

from typing import Dict

from .base import ModelType
from .vllm import VllmType
from .vllm_legacy import VllmLegacyType

TYPES: Dict[str, ModelType] = {t.name: t for t in (VllmType(), VllmLegacyType())}

__all__ = ["TYPES", "ModelType"]
