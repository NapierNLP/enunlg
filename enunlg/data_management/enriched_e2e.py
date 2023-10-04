import os
from typing import Iterable, List, Optional, Union

import box
import xsdata.formats.dataclass.parsers as xsparsers

from enunlg.formats.xml.enriched_e2e import Entries
import enunlg.data_management.iocorpus


# TODO add hydra configuration for enriched e2e stuff
ENRICHED_E2E_CONFIG = box.Box({'ENRICHED_E2E_DIR': os.path.join(os.path.dirname(__file__), '../../datasets/raw/EnrichedE2E/')})

E2E_SPLIT_DIRS = ('train', 'dev', 'test')


class EnrichedE2ECorpus(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[Iterable] = None, filename_or_list: Optional[Union[str, List[str]]] = None):
        if seq is None:
            seq = []
        super().__init__(seq)
        self.entries = []
        if filename_or_list is not None:
            if isinstance(filename_or_list, list):
                for filename in filename_or_list:
                    print(filename)
                    self.load_file(filename)
            elif isinstance(filename_or_list, str):
                self.load_file(filename_or_list)
            else:
                raise TypeError(f"Expected filename_or_list to be None, str, or list, not {type(filename_or_list)}")

    def load_file(self, filename):
        entries_object = xsparsers.XmlParser().parse(filename, Entries)
        self.entries.extend(entries_object.entry)


def load_enriched_e2e(splits: Optional[Iterable[str]] = None, enriched_e2e_config: Optional[box.Box] = None) -> EnrichedE2ECorpus:
    """

    :param splits: which splits to load
    :param enriched_e2e_config: a box.Box or omegaconf.DictConfig like object containing the basic
                                information about the e2e corpus to be used
    :return: the corpus of MR-text pairs with metadata
    """
    if enriched_e2e_config is None:
        enriched_e2e_config = ENRICHED_E2E_CONFIG
    corpus_name = "E2E Challenge Corpus"
    default_splits = E2E_SPLIT_DIRS
    data_directory = enriched_e2e_config.ENRICHED_E2E_DIR
    if splits is None:
        splits = default_splits
    elif not set(splits).issubset(default_splits):
        raise ValueError(f"`splits` can only contain a subset of {default_splits}. Found {splits}.")
    fns = []
    for split in splits:
        print(split)
        fns.extend([os.path.join(data_directory, split, fn) for fn in os.listdir(os.path.join(data_directory, split))])

    corpus = EnrichedE2ECorpus(filename_or_list=fns)
    corpus.metadata = {'name': corpus_name,
                       'splits': splits,
                       'directory': data_directory}
    return corpus
