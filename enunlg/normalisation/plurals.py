from typing import Callable, Optional

import logging

import bidict

logger = logging.getLogger(__name__)


class TGenPluraliser(object):
    def __init__(self):
        self._sg_to_pl = bidict.MutableBidict({None: None,
                                               'child': 'children'})

    def get_plural(self, sg_form: str, default: Optional[Callable] = None):
        if sg_form not in self._sg_to_pl:
            if default is None:
                raise ValueError(f"{sg_form} is not in our dictionary and no default pluralising function is provided")
            else:
                # For example, lambda x: f"{x}s"
                self._sg_to_pl[sg_form] = default(sg_form)
        return self._sg_to_pl[sg_form]

    def get_singular(self, pl_form: str, default: Optional[Callable] = lambda x: x.strip('s') if x.endswith('s') else x):
        if pl_form not in self._sg_to_pl.inverse:
            if default is None:
                raise ValueError(f"{pl_form} is not in our dictionary and no default depluralising function is provided")
            else:
                self._sg_to_pl[default(pl_form)] = pl_form
        return self._sg_to_pl.inverse[pl_form]
