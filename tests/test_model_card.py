# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Generated model cards keep model-authored quickstarts with the package."""

from tt_model.hub import render_model_card


def test_laguna_card_has_pool_quickstart(laguna):
    card = render_model_card(laguna, ["models/"])

    assert "## Quickstart" in card
    assert "### Use with the pool coding agent" in card
    assert 'POOLSIDE_STANDALONE_BASE_URL="http://127.0.0.1:8000"' in card
    assert 'POOLSIDE_STANDALONE_MODEL="poolside/Laguna-XS-2.1"' in card
    assert card.index("## Quickstart") < card.index("## Serve profiles")


def test_card_quickstart_is_optional(ornith):
    card = render_model_card(ornith, [])

    assert "## Quickstart" in card
    assert "POOLSIDE_STANDALONE_BASE_URL" not in card
