# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Guard against the `web/config.js` catalog/bundle tags drifting from the Python
constants that actually write them.

The catalog is a pure index: `web/config.js` queries the Hub for repos carrying the
tags that `tt-model publish` / `tt-model push` write. If the string literals in the JS
drift from `tt_kernel.TT_MODEL_CATALOG_TAG` / `TT_MODEL_TAG`, every published bundle
becomes invisible (issue #42 part 1). This test pins them together.
"""

import pathlib
import re

from tt_kernel import TT_MODEL_CATALOG_TAG, TT_MODEL_TAG

CONFIG_JS = pathlib.Path(__file__).resolve().parent.parent / "web" / "config.js"


def _extract(key: str) -> str:
    text = CONFIG_JS.read_text()
    match = re.search(rf'{key}:\s*"([^"]+)"', text)
    assert match is not None, f"{key} not found in {CONFIG_JS}"
    return match.group(1)


def test_catalog_tag_matches_constant():
    assert _extract("CATALOG_TAG") == TT_MODEL_CATALOG_TAG


def test_bundle_tag_matches_constant():
    assert _extract("BUNDLE_TAG") == TT_MODEL_TAG
