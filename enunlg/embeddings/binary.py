from ast import literal_eval
from typing import Iterable, List, MutableMapping, Optional, Set, Text, Tuple, TYPE_CHECKING

import logging
import os
import tarfile

import bidict
import omegaconf

if TYPE_CHECKING:
    import enunlg.meaning_representation.dialogue_acts as dialogue_acts

logger = logging.getLogger(__name__)


class DialogueActEmbeddings(object):
    STATE_ATTRIBUTES = ("_collapse_values", "_acts", "_slot_value_pairs", "_unk_sv_pair", "acts", "slot_value_pairs")

    def __init__(self, multi_da_seq: Iterable["dialogue_acts.MultivaluedDA"], collapse_values=True) -> None:
        """Designed to create embeddings compatible with SC-LSTM."""
        self._collapse_values = collapse_values
        self._dimensionality = None
        self._acts: Set[str] = set()
        # TODO check how to specify bidict.OrderedBidict as the type with restrictions on its contents
        # We need to update this hint bc self.acts (and .slot_value_pairs) both need to have .inv to work
        self.acts: MutableMapping[str, int] = bidict.OrderedBidict()
        self._slot_value_pairs: Set[Tuple[str, Optional[str]]] = set()
        self.slot_value_pairs: MutableMapping[Tuple[str, Optional[str]], int] = bidict.OrderedBidict()
        self._initialise_embeddings(multi_da_seq)
        self._unk_sv_pair = 0

    @property
    def collapse_values(self):
        return self._collapse_values

    @property
    def dimensionality(self):
        if self._dimensionality is None:
            self._dimensionality = len(self._acts) + len(self._slot_value_pairs)
        return self._dimensionality

    @property
    def size(self):
        return self.dimensionality

    @property
    def dialogue_act_size(self):
        return len(self._acts)

    @property
    def slot_value_size(self):
        return len(self._slot_value_pairs)

    def _slot_value_decoder(self, slot_value_box: MutableMapping[str, List]) -> List[Tuple[str, Optional[str]]]:
        retval: List[Tuple[str, Optional[str]]] = []
        for slot in slot_value_box:
            if isinstance(slot_value_box[slot], Text):
                raise ValueError("slot_value_decoder only works on iterables containing strings, not bare strings")
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
        self._unk_sv_pair = len(self.slot_value_pairs) + 1

    def embed_da(self, multi_da: "dialogue_acts.MultivaluedDA") -> List[float]:
        act_embedding = [0.0 for _ in range(len(self.acts))]
        act_embedding[self.acts[multi_da.act_type]] = 1.0

        slot_value_embedding = [0.0 for _ in range(len(self.slot_value_pairs))]
        slot_values = self._slot_value_decoder(multi_da.slot_values)
        for slot_value in slot_values:
            slot_value_embedding[self.slot_value_pairs.get(slot_value, self._unk_sv_pair)] = 1.0
        act_embedding.extend(slot_value_embedding)
        return act_embedding

    def embedding_to_string(self, embedding):
        act_embedding, slot_value_embedding = embedding[:len(self.acts)], embedding[len(self.acts):]
        retval = [self.acts.inv[act_embedding.index(1.0)]]
        slot_value_indices = [index for index, sv in enumerate(slot_value_embedding) if sv != 0.0]
        for slot_value_index in slot_value_indices:
            retval.append(str(self.slot_value_pairs.inv[slot_value_index]))
        return " ".join(retval)

    def _save_classname_to_dir(self, directory_path):
        with open(os.path.join(directory_path, "__class__.__name__"), 'w') as class_file:
            class_file.write(self.__class__.__name__)

    def save(self, filepath, tgz=False):
        os.mkdir(filepath)
        self._save_classname_to_dir(filepath)
        state = {}
        for attribute in self.STATE_ATTRIBUTES:
            curr_obj = getattr(self, attribute)
            if attribute in ("acts", "slot_value_pairs"):
                # These are bidicts, so we'll save them as dicts
                state[attribute] = {str(k): curr_obj[k] for k in curr_obj}
            else:
                if isinstance(curr_obj, set):
                    state[attribute] = tuple(curr_obj)
                else:
                    state[attribute] = curr_obj
        with open(os.path.join(filepath, "_save_state.yaml"), 'w') as state_file:
            omegaconf.OmegaConf.save(state, state_file)
        if tgz:
            with tarfile.open(f"{filepath}.tgz", mode="x:gz") as out_file:
                out_file.add(filepath, arcname=os.path.basename(filepath))

    @classmethod
    def load_from_dir(cls, filepath):
        with open(os.path.join(filepath, '__class__.__name__'), 'r') as class_file:
            assert class_file.read().strip() == cls.__name__
        new_instance = cls([])
        state = omegaconf.OmegaConf.load(os.path.join(filepath, "_save_state.yaml"))
        for attribute in state:
            setattr(new_instance, attribute, state[attribute])
        # These are saved as tuples so we need to convert them to the correct type
        new_instance._acts = set(new_instance._acts)
        new_instance._slot_value_pairs = set(new_instance._slot_value_pairs)
        # These are saved as plain dicts, so we need to convert them to the right type
        new_instance.acts = bidict.OrderedBidict(new_instance.acts)
        strings = list(new_instance.slot_value_pairs.keys())
        new_instance.slot_value_pairs = dict(new_instance.slot_value_pairs)
        for k in strings:
            new_instance.slot_value_pairs[literal_eval(k)] = new_instance.slot_value_pairs[k]
            new_instance.slot_value_pairs.pop(k)
        new_instance.slot_value_pairs = bidict.OrderedBidict(new_instance.slot_value_pairs)
        return new_instance
