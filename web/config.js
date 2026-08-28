// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

// Deployment configuration for the tt-kernel community catalog.
//
// The catalog is a PURE INDEX. Everything below is read live from the Hugging Face Hub
// public API by the visitor's browser — this site hosts and stores nothing. Bundles are
// owned and governed by whoever pushed them; the catalog only shows a pointer to their
// public HF repo. To self-host, drop this folder on any static web server (see README.md).
window.TTK_CONFIG = {
  // HF Hub origin. Point at a mirror if you run one.
  HF_ORIGIN: "https://huggingface.co",

  // The opt-in tag a bundle must carry to appear here. Set by `tt-kernel push --publish`
  // or `tt-kernel publish`. Must match tt_kernel.TT_MODEL_CATALOG_TAG.
  CATALOG_TAG: "tt-model-catalog",

  // The base bundle tag, and the manifest filename inside each repo. Must match
  // tt_kernel.TT_MODEL_TAG.
  BUNDLE_TAG: "tt-model-cache",
  MANIFEST_NAME: "tt_kernel_manifest.json",

  // Arch tags a bundle may carry (used to offer arch filters without a manifest fetch).
  KNOWN_ARCHES: ["blackhole", "wormhole_b0", "wormhole", "grayskull"],

  // Model-capability tags a producer sets via `tt-kernel push --capability <key>`. The
  // catalog reads these straight from the repo's tags (no manifest fetch), rendering them
  // as badges and dynamic filter chips. Map each recognized tag to its display label;
  // aliases can share a label. Add rows here as new capability kernels land.
  CAPABILITIES: {
    "moe": "MoE",
    "mixture-of-experts": "MoE",
    "sliding-window-attention": "Sliding window",
    "swa": "Sliding window",
  },

  // Max repos to pull from the list endpoint, and how many manifests to enrich at once.
  LIST_LIMIT: 1000,
  ENRICH_CONCURRENCY: 6,

  // Cosmetic: title/brand shown in the header.
  BRAND: "tt-kernel catalog",
  TAGLINE: "Precompiled Tenstorrent kernel bundles, published by the community.",
};
