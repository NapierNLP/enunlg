from dataclasses import dataclass
from typing import Iterable, Tuple

import logging

import box

logger = logging.getLogger(__name__)


@dataclass
class DialogueAct:
    act_type: str
    slot_values: box.Box

    def __init__(self, act_type, slot_values):
        self.act_type = act_type
        self.slot_values = box.Box(slot_values)


class InformDA(DialogueAct):
    def __init__(self, slot_values):
        super(InformDA, self).__init__("Inform", slot_values)


class MultivaluedDA(DialogueAct):
    def __init__(self, act_type, slot_values):
        """Represents dialogue acts where individual slots can take on multiple values

        Assumes that values are stored as lists
        """
        super(MultivaluedDA, self).__init__(act_type, slot_values)

    def __len__(self):
        return sum([len(self.slot_values[slot]) for slot in self.slot_values])

    def __hash__(self):
        return hash((self.act_type, box.Box(self.slot_values, frozen_box=True)))

    @staticmethod
    def from_slot_value_list(act_type, slot_values: Iterable[Tuple[str, str]]) -> "MultivaluedDA":
        """Create a MultivalueDA when we have an iterable of slot-value pairs rather than a ready-to-go Box."""
        slot_value_box = box.Box(default_box=True, default_box_attr=list)
        for slot, value in slot_values:
            if slot in slot_value_box:
                slot_value_box[slot].append(value)
            else:
                slot_value_box[slot] = [value]
        return MultivaluedDA(act_type, slot_value_box)
