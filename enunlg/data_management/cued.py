from pathlib import Path
from typing import Dict, Iterable, Optional

import itertools
import json
import logging

import box
import regex

import enunlg.data_management.iocorpus as iocorpus
import enunlg.meaning_representation.dialogue_acts as da_lib

logger = logging.getLogger(__name__)

SFX_RESTAURANT_DIR = Path(__file__).parent / '../../datasets/raw/RNNLG/data/original/restaurant'
SFX_HOTEL_DIR = Path(__file__).parent / '../../datasets/raw/RNNLG/data/original/hotel'
LAPTOP_DIR = Path(__file__).parent / '../../datasets/raw/RNNLG/data/original/laptop'
TV_DIR = Path(__file__).parent / '../../datasets/raw/RNNLG/data/original/tv'

WEN_ET_AL_DATASETS: Dict[str, Path] = {"sfx_restaurant": SFX_RESTAURANT_DIR,
                                       "sfx_hotel": SFX_HOTEL_DIR,
                                       "laptop": LAPTOP_DIR,
                                       "tv": TV_DIR}

CUED_SPLITS = ('train', 'valid', 'test')

# From RNNLG/resources/special_values.txt
SCLSTM_SPECIAL = {
    "none": ["none"],
    "yes": ["true", "yes"],
    "no": ["false", "no"],
    "dontcare": ["dontcare", "dont_care"]}

SCLSTM_SPECIAL_VALUES = set(list(SCLSTM_SPECIAL.keys()) + ['?'])

ALL_FIELDS = {"address", "area", "count", "food", "goodformeal", "kidsallowed", "name", "near", "phone", "postcode", "price", "pricerange", "type"}


class CUEDPair(iocorpus.IOPair):
    mr: da_lib.DialogueAct
    text: str


class CUEDCorpus(iocorpus.IOCorpus):
    def __init__(self, seq: Iterable[CUEDPair]):
        super().__init__(seq)

    def print_summary_stats(self):
        print(f"{self.metadata=}")
        print(f"num entries: {len(self)}")
        mr_lengths = []
        mr_types = set()
        text_lengths = []
        text_types = set()
        for item in self:
            mr_lengths.append(len(item.mr))
            mr_types.add(item.mr)
            text_lengths.append(len(item.text.split()))
            text_types.add(tuple(item.text.split()))
        
        print(f"MRs:\t\t{sum(mr_lengths)/len(mr_lengths):.2f} [{min(mr_lengths)},{max(mr_lengths)}]")
        print(f"    with {len(mr_types)} types across {len(mr_lengths)} tokens.")
        print("NB: these values don't tokenize MRs")

        print(f"Texts:\t\t{sum(text_lengths)/len(text_lengths):.2f} [{min(text_lengths)},{max(text_lengths)}]")
        print(f"    with {len(text_types)} types across {len(text_lengths)} tokens.")

def parse_cued_dialogue_acts(dialogue_act_string, keep_values=False):
    """Adapted from RNNLG/loader/data_reader.py DialogueActParser.parse()"""
    act_type = dialogue_act_string.split('(')[0]
    slt2vals = dialogue_act_string.split('(')[1].replace(')', '').split(';')
    slot_value_list = []
    for slt2val in slt2vals:
        if slt2val == '':  # no slot
            slot_value_list.append((None, None))
        elif '=' not in slt2val:  # no value
            slt2val = slt2val.replace('_', '').replace(' ', '')
            slot_value_list.append((slt2val.strip('\'\"'), '?'))
        else:  # both slot and value exist
            s, v = [x.strip('\'\"') for x in slt2val.split('=')]
            s = s.replace('_', '').replace(' ', '')
            for key, vals in SCLSTM_SPECIAL.items():
                if v in vals:  # unify the special values
                    v = key
            if v not in SCLSTM_SPECIAL and not keep_values:  # delexicalisation
                v = '_'
            slot_value_list.append((s, v))
    return da_lib.MultivaluedDA.from_slot_value_list(act_type, slot_value_list)


def delexicalise_exact_matches(pair: CUEDPair, fields_to_delex: Optional[Iterable] = None, with_subscripts: bool = True) -> CUEDPair:
    if fields_to_delex:
        new_mr = pair.mr.slot_values.to_dict()
        # Insert spaces around the text
        new_text = f" {pair.text} "
        for field in fields_to_delex:
            if field in new_mr and field not in SCLSTM_SPECIAL_VALUES:
                new_value = []
                for index, value in enumerate(new_mr[field], start=1):
                    # <based on SC-LSTM exact match formatter>
                    vs = value.replace(' or ', ' and ').split(' and ')
                    permutations = set([' and '.join(x) for x in itertools.permutations(vs)] +
                                       [' or '.join(x) for x in itertools.permutations(vs)])
                    for permutation in permutations:
                        if permutation in new_text:
                            if with_subscripts:
                                replacement = f"SLOT_{field.upper()}_{index}"
                            else:
                                replacement = f"SLOT_{field.upper()}"
                            if permutation != '?':
                                new_text = regex.sub(permutation, replacement, new_text)
                                new_value.append(replacement)
                                break
                    # </based on>
                new_mr[field] = new_value
        return CUEDPair(da_lib.MultivaluedDA(pair.mr.act_type, box.Box(new_mr, frozen_box=True)), new_text)
    else:
        return pair


def multivalued_da_to_cued_mr_string(multida: da_lib.MultivaluedDA, combine_multi_valued_slots=False):
    """
    Create a string representation in Wen et al.'s format.

    NB: Not every string value in dialogue acts is escaped in their corpora, and some values have multiple valid ways
    of encoding them. The output here provides a sensible standardised output format.
    :param multida: the dialogue act to be converted
    :param combine_multi_valued_slots: whether we want to combine multiple values with " or " as a single value
    """
    fields = []
    for slot in multida.slot_values:
        if slot is not None:
            if combine_multi_valued_slots:
                value_string = " or ".join(list(multida.slot_values[slot]))
                if value_string == "?":
                    fields.append(slot)
                    continue
                if " " in value_string or all(c.isdigit() for c in value_string):
                    if "'" in value_string:
                        value_string = f'"{value_string}"'
                    else:
                        value_string = f"'{value_string}'"
                fields.append(f"{slot}={value_string}")
            else:
                for value_string in multida.slot_values[slot]:
                    if value_string == "?":
                        fields.append(slot)
                        continue
                    if " " in value_string or all(c.isdigit() for c in value_string):
                        if "'" in value_string:
                            value_string = f'"{value_string}"'
                        else:
                            value_string = f"'{value_string}'"
                    fields.append(f"{slot}={value_string}")
    return f"{multida.act_type}({';'.join(fields)})"


def load_cued_json(filepath, includes_comment_header=True):
    """Load a JSON file, potentially ignoring the 5 line comment header used by Wen et al."""
    with Path(filepath).open() as input_file:
        if includes_comment_header:
            for _ in range(5):
                next(input_file)
        return json.load(input_file)


def load_cued_data(filepath, includes_comment_header=True, human_ref_only=True):
    data = load_cued_json(filepath, includes_comment_header)
    if human_ref_only:
        return CUEDCorpus([CUEDPair(parse_cued_dialogue_acts(da, keep_values=True), human_ref)
                           for da, human_ref, _ in data])
    else:
        # TODO create a way to load triples or to alternatively load the handcrafted-rule-based reference sents.
        message = "No implementation for Wen et al. triples yet."
        raise NotImplementedError(message)


def load_wen_et_al_dataset(name: str, splits=None):
    data_directory = WEN_ET_AL_DATASETS.get(name)
    if data_directory is None:
        message = f"`name` can only be one of {list(WEN_ET_AL_DATASETS.keys())}. Got {name}"
        raise ValueError(message)
    if splits is None:
        splits = CUED_SPLITS
    elif not set(splits).issubset(CUED_SPLITS):
        message = f"`splits` can only contain a subset of {CUED_SPLITS}. Found {splits}."
        raise ValueError(message)
    corpus = CUEDCorpus([])
    for split in splits:
        corpus.extend(load_cued_data(Path(data_directory) / f'{split}.json'))
    corpus.metadata = {'name': name,
                       'splits': splits,
                       'directory': data_directory}
    return corpus

def load_sfx_restaurant(splits=None):
    return load_wen_et_al_dataset("sfx_restaurant", splits)


def load_sfx_hotel(splits=None):
    return load_wen_et_al_dataset("sfx_hotel", splits)


def load_wen_laptops(splits=None):
    return load_wen_et_al_dataset("laptop", splits)


def load_wen_tv(splits=None):
    return load_wen_et_al_dataset("tv", splits)


if __name__ == "__main__":
    corpus = load_sfx_restaurant()
    print(len(corpus))
    for da, human_ref in corpus:
        print(da)
        print(human_ref)
