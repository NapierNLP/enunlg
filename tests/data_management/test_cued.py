import os

import pytest

import enunlg.data_management.cued as cued

SFX_JSON = cued.load_cued_json(os.path.join(cued.SFX_RESTAURANT_DIR, 'train.json'))


@pytest.mark.parametrize('inp, expected', zip(cued.load_sfx_restaurant(['train']), [x[0] for x in SFX_JSON]))
def test_load_sfx_restaurant(inp, expected):
    """
    Confirm that our MultiDA representations correspond to the data in the sfx restaurant corpus

    The SFX restaurant corpus is inconsistent in the quotation of some string values and whether or not underscores are present for dontcare values.
    """
    assert cued.multivalued_da_to_cued_mr_string(inp.mr).replace("_", "").replace("'", "") == expected.replace("_", "").replace("'", "")

