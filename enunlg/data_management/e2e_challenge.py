import logging
from typing import Iterable, List, Optional, Tuple

import csv
import os

import box
import omegaconf
import regex

from enunlg.meaning_representation.slot_value import SlotValueMR

import enunlg.data_management.iocorpus as iocorpus

logger = logging.getLogger(__name__)

# TODO add hydra configuration for e2e challenge stuff!
E2E_DIR = os.path.join(os.path.dirname(__file__), '../../datasets/e2e-dataset/')
E2E_CLEANED_DIR = os.path.join(os.path.dirname(__file__), '../../datasets/e2e-cleaning/cleaned-data/')

E2E_CONFIG = omegaconf.DictConfig({'E2E_DIR': os.path.join(os.path.dirname(__file__),
                                                           '../../datasets/raw/e2e-dataset/'),
                                   'E2E_CLEANED_DIR': os.path.join(os.path.dirname(__file__),
                                                                   '../../datasets/raw/e2e-cleaning/cleaned-data/')})

E2E_SPLITS = ('trainset', 'devset', 'testset_w_refs')
E2E_CLEANED_SPLITS = ('train-fixed.no-ol', 'devel-fixed.no-ol', 'test-fixed')

# It is necessary to allow spaces in order to cover e2e cleaned data,
# but not for the original e2e dataset.
MR_LABEL_CHARS = "[A-Za-z_ ]+"
MR_DELIM_CHARS = "[, ]"
MR_BRA = r"\["
MR_KET = r"\]"
MR_VALUE_CHARS = r"[A-Za-z0-9_\-,$£ é]+"


class E2EPair(iocorpus.IOPair):
    mr: SlotValueMR

    def sort_mr(self, in_place: bool = True) -> Optional[dict]:
        sorted_mr = SlotValueMR({key: self.mr[key] for key, _ in sorted(self.mr.items())})
        if in_place:
            self.mr = sorted_mr
        else:
            return sorted_mr
        # Added for PEP8 consistency and mypy happiness
        return None


class E2ECorpus(iocorpus.IOCorpus):
    def __init__(self, seq: Iterable[E2EPair]):
        super().__init__(seq)

    def sort_mr_elements(self):
        for x in self:
            x.sort_mr()

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


def parse_mr(e2e_mr: str) -> SlotValueMR:
    facts = {}
    loop_check = 0
    while e2e_mr:
        if loop_check > 100:
            logging.warning(f"Regexes for MRs cannot parse: {e2e_mr}")
            break
        e2e_mr = e2e_mr.strip()
        mo = regex.match(f"^({MR_LABEL_CHARS}){MR_BRA}({MR_VALUE_CHARS}){MR_KET}", e2e_mr)
        if mo:
            facts[mo.group(1)] = mo.group(2)
            e2e_mr = regex.sub(MR_DELIM_CHARS, '', e2e_mr[mo.span(0)[1]:])
        loop_check += 1
    return SlotValueMR(facts)


def delexicalise_exact_matches(pair: E2EPair, fields_to_delex: Optional[Iterable] = None) -> E2EPair:
    if fields_to_delex is None:
        return pair
    else:
        new_mr = pair.mr.to_dict()
        new_text = pair.text
        for field in fields_to_delex:
            if field in new_mr:
                field_with_spaces = "".join([(" "+i if i.isupper() else i) for i in new_mr[field]]).strip()
                if new_mr[field] in new_text:
                    replacement = f"__{field.upper()}__"
                    new_text = regex.sub(new_mr[field], replacement, new_text)
                    new_mr[field] = replacement
                elif field_with_spaces in new_text:
                    replacement = f"__{field.upper()}__"
                    new_text = regex.sub(field_with_spaces, replacement, new_text)
                    new_mr[field] = replacement
        return E2EPair(SlotValueMR(new_mr), new_text)


def load_e2e_csv(filepath: str) -> List[Tuple[str, str]]:
    """E2E CSV files' first column is MRs and the second column is texts. There is always a header line."""
    with open(filepath, 'r') as in_file:
        csv_reader = csv.reader(in_file)
        next(csv_reader)
        return E2ECorpus([E2EPair(parse_mr(pair[0]), pair[1]) for pair in csv_reader])


def load_e2e(corpus_config: omegaconf.DictConfig,
             splits: Optional[Iterable[str]]) -> E2ECorpus:
    """
    :return: the corpus of MR-text pairs with metadata
    """
    default_splits = set(corpus_config.splits.keys())
    if not set(splits).issubset(default_splits):
        message = f"`splits` can only contain a subset of {default_splits}. Found {splits}."
        raise ValueError(message)
    corpus = E2ECorpus([])
    for split in splits:
        for csv_file in corpus_config.splits[split]:
            corpus.extend(load_e2e_csv(os.path.join(os.path.dirname(__file__), corpus_config.load_dir, f"{csv_file}")))
    corpus.metadata = {'name': corpus_config.display_name,
                       'splits': splits,
                       'directory': corpus_config.load_dir}
    return corpus
