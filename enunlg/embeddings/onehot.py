from typing import Dict, Iterable, List, Text, TYPE_CHECKING

import bidict

if TYPE_CHECKING:
    import enunlg.meaning_representation.dialogue_acts as dialogue_acts


class DialogueActEmbeddings(object):
    def __init__(self, multi_da_seq: Iterable["dialogue_acts.MultivaluedDA"], collapse_values=True) -> None:
        """Designed to create embeddings compatible with SC-LSTM."""
        self._collapse_values = collapse_values
        self._dimensionality = None
        self._acts = set()
        self.acts: Dict[str, int] = bidict.OrderedBidict()
        self._slot_value_pairs = set()
        self.slot_value_pairs = bidict.OrderedBidict()
        self._initialise_embeddings(multi_da_seq)

    @property
    def collapse_values(self):
        return self._collapse_values

    @property
    def dimensionality(self):
        if self._dimensionality is None:
            self._dimensionality = len(self._acts) + len(self._slot_value_pairs)
        return self._dimensionality

    @property
    def dialogue_act_size(self):
        return len(self._acts)

    @property
    def slot_value_size(self):
        return len(self._slot_value_pairs)

    def _slot_value_decoder(self, slot_value_box):
        retval = []
        for slot in slot_value_box:
            if isinstance(slot_value_box[slot], Text):
                raise ValueError(f"slot_value_decoder only works on iterables containing strings, not bare strings")
            if not isinstance(slot_value_box[slot], Iterable):
                raise ValueError(f"Expected list but got {slot_value_box[slot]=}")
            for index, value in enumerate(slot_value_box[slot], start=1):
                if value in {'none', 'yes', 'no', 'dontcare', '?'}:
                    retval.append((slot, f"{value}"))
                elif value is None:
                    retval.append((slot, value))
                else:
                    if self.collapse_values:
                        retval.append((slot, f"_{index}"))
                    else:
                        retval.append((slot, value))
        return retval

    def _initialise_embeddings(self, seq: Iterable["dialogue_acts.MultivaluedDA"]) -> None:
        for da in seq:
            self._acts.add(da.act_type)
            self._slot_value_pairs.update(set(self._slot_value_decoder(da.slot_values)))
        for index, act in enumerate(self._acts):
            self.acts[act] = index
        for index, slot_value in enumerate(self._slot_value_pairs):
            self.slot_value_pairs[slot_value] = index

    def embed_da(self, multi_da: "dialogue_acts.MultivaluedDA") -> List[float]:
        act_embedding = [0.0 for _ in range(len(self.acts))]
        act_embedding[self.acts[multi_da.act_type]] = 1.0

        slot_value_embedding = [0.0 for _ in range(len(self.slot_value_pairs))]
        slot_values = self._slot_value_decoder(multi_da.slot_values)
        for slot_value in slot_values:
            slot_value_embedding[self.slot_value_pairs[slot_value]] = 1.0
        act_embedding.extend(slot_value_embedding)
        return act_embedding

    def embedding_to_string(self, embedding):
        act_embedding, slot_value_embedding = embedding[:len(self.acts)], embedding[len(self.acts):]
        retval = [self.acts.inv[act_embedding.index(1.0)]]
        slot_value_indices = [index for index, sv in enumerate(slot_value_embedding) if sv != 0.0]
        for slot_value_index in slot_value_indices:
            retval.append(str(self.slot_value_pairs.inv[slot_value_index]))
        return " ".join(retval)
