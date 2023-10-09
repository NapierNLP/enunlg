from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Union

import difflib
import logging
import os
import random

import omegaconf
import xsdata.formats.dataclass.parsers as xsparsers

from enunlg.formats.xml.enriched_e2e import EnrichedE2EEntries
from enunlg.meaning_representation.slot_value import SlotValueMR
import enunlg.data_management.iocorpus
import enunlg.data_management.pipelinecorpus

# TODO add hydra configuration for enriched e2e stuff
ENRICHED_E2E_CONFIG = omegaconf.DictConfig({'ENRICHED_E2E_DIR': os.path.join(os.path.dirname(__file__),
                                                                             '../../datasets/raw/EnrichedE2E/')})

E2E_SPLIT_DIRS = ('train', 'dev', 'test')

DELEX_LABELS = ["__AREA__", "__CUSTOMER_RATING__", "__EATTYPE__", "__FAMILYFRIENDLY__", "__FOOD__", "__NAME__",
                "__NEAR__", "__PRICERANGE__"]

DIFFER = difflib.Differ()


def extract_reg_from_template_and_text(text, template):
    diff = DIFFER.compare(text, template)
    keys = []
    values = []
    curr_add = []
    curr_min = []
    for x in diff:
        if x.startswith('-'):
            curr_min.append(x[2])
        elif x.startswith('+'):
            curr_add.append(x[2])
        else:
            if curr_min:
                values.append("".join(curr_min))
                curr_min = []
            if curr_add:
                keys.append("".join(curr_add))
                curr_add = []
    result_dict = defaultdict(list)
    for key, value in zip(keys, values):
        result_dict[key].append(value)
    return result_dict

class EnrichedE2ECorpusRaw(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[Iterable] = None, filename_or_list: Optional[Union[str, List[str]]] = None):
        if seq is None:
            seq = []
        super().__init__(seq)
        self.entries = []
        if filename_or_list is not None:
            if isinstance(filename_or_list, list):
                for filename in filename_or_list:
                    logging.info(filename)
                    self.load_file(filename)
            elif isinstance(filename_or_list, str):
                self.load_file(filename_or_list)
            else:
                raise TypeError(f"Expected filename_or_list to be None, str, or list, not {type(filename_or_list)}")
            self.extend(self.entries)

    def load_file(self, filename):
        entries_object = xsparsers.XmlParser().parse(filename, EnrichedE2EEntries)
        self.entries.extend(entries_object.entries)


class PipelineCorpusMapper(object):
    def __init__(self, input_format, output_format, annotation_layer_mappings: Dict[str, Callable]):
        """
        Create a function which will map from `input_format` to `output_format` using `annotation_layer_mappings`.
        """
        self.input_format = input_format
        self.output_format = output_format
        self.annotation_layer_mappings = annotation_layer_mappings

    def __call__(self, input_corpus: Iterable) -> List:
        logging.debug(f'successful call to {self.__class__.__name__} as a function (rather than a class)')
        if isinstance(input_corpus, self.input_format):
            logging.debug('passed the format check')
            output_seq = []
            for entry in input_corpus:
                output = []
                for layer in self.annotation_layer_mappings:
                    logging.debug(f"processing {layer}")
                    output.append(self.annotation_layer_mappings[layer](entry))
                # EnrichedE2E-formated datasets have up to N distinct targets for each single input
                # This will show up as the first 'layer' having length 1 and subsequent layers having length > 1
                num_targets = max([len(x) for x in output])
                # We expand any layers of length 1, duplicating their entries, and preserving the rest of the layers
                output = [x * num_targets if len(x) == 1 else x for x in output]
                assert all([len(x) == num_targets for x in output]), f"expected all layers to have the same number of items, but received: {[len(x) for x in output]}"
                # For each of the N distinct targets, create a self.outputformat object and append it to the output_seq
                for i in range(num_targets-1):
                    item = self.output_format({key: output[idx][i] for idx, key in enumerate(self.annotation_layer_mappings.keys())})
                    output_seq.append(item)
                logging.debug(f"Num entries so far: {len(output_seq)}")
            return output_seq
        else:
            raise TypeError(f"Cannot run {self.__class__} on {type(input_corpus)}")


class EnrichedE2EItem(object):
    def __init__(self, layers: dict):
        sanitized_layer_names = [layer_name.replace("-", "_") for layer_name in layers.keys()]
        self.layers = sanitized_layer_names
        for new_name, layer in zip(sanitized_layer_names, layers):
            self.__setattr__(new_name, layers[layer])

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __repr__(self):
        attr_string = ", ".join([f"{layer}={str(self[layer])}" for layer in self.layers])
        return f"{self.__class__.__name__}({attr_string})"


class EnrichedE2ECorpus(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: List[EnrichedE2EItem]):
        if seq:
            layer_names = seq[0].layers
            assert all([item.layers == layer_names for item in seq]), f"Expected all items in seq to have the layers: {layer_names}"
            self.layers = layer_names
        super(EnrichedE2ECorpus, self).__init__(seq)

    @property
    def views(self):
        return [(l1, l2) for l1, l2 in zip(self.layers, self.layers[1:])]

    def items_by_view(self, view):
        for item in self:
            yield item[view[0]], item[view[1]]


def extract_raw_input(entry):
    mr = {}
    for input in entry.source.inputs:
        mr[input.attribute] = input.value
    return [SlotValueMR(mr, frozen_box=True)]


def extract_selected_input(entry):
    targets = []
    for target in entry.targets:
        mr = {}
        for sentence in target.structuring.sentences:
            for input in sentence.content:
                mr[input.attribute] = input.value
        targets.append(SlotValueMR(mr, frozen_box=True))
    return targets


def extract_ordered_input(entry):
    targets = []
    for target in entry.targets:
        selected_inputs = []
        for sentence in target.structuring.sentences:
            mr = {}
            for input in sentence.content:
                mr[input.attribute] = input.value
            selected_inputs.append(SlotValueMR(mr, frozen_box=True))
        targets.append(tuple(selected_inputs))
    return targets


def extract_sentence_segmented_input(entry):
    targets = []
    for target in entry.targets:
        selected_inputs = []
        for sentence in target.structuring.sentences:
            mr = {}
            for input in sentence.content:
                mr[input.attribute] = input.value
            selected_inputs.append(SlotValueMR(mr, frozen_box=True))
        targets.append(tuple(selected_inputs))
    return targets


def extract_lexicalization(entry):
    return [target.lexicalization for target in entry.targets]


def extract_reg_completed_lex_random(entry):
    texts = [target.text for target in entry.targets]
    templates = [target.template for target in entry.targets]
    texts_templates = zip(texts, templates)
    lexes = [target.lexicalization for target in entry.targets]
    regged_lexes = []
    for pair, lex in zip(texts_templates, lexes):
        reg_dict = extract_reg_from_template_and_text(pair[0], pair[1])
        for key in reg_dict:
            lex = lex.replace(key, random.choice(reg_dict[key]))
        regged_lexes.append(lex)
    return regged_lexes


def extract_raw_output(entry):
    return [target.text for target in entry.targets]


def load_enriched_e2e(splits: Optional[Iterable[str]] = None, enriched_e2e_config: Optional[SlotValueMR] = None) -> EnrichedE2ECorpus:
    """

    :param splits: which splits to load
    :param enriched_e2e_config: a SlotValueMR or omegaconf.DictConfig like object containing the basic
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
        logging.info(split)
        fns.extend([os.path.join(data_directory, split, fn) for fn in os.listdir(os.path.join(data_directory, split))])

    corpus = EnrichedE2ECorpusRaw(filename_or_list=fns)
    corpus.metadata = {'name': corpus_name,
                       'splits': splits,
                       'directory': data_directory}
    logging.info(len(corpus))
    #lambda entry: [x.sentence.content for x in entry.targets.structuring]

    enriched_e2e_factory = PipelineCorpusMapper(EnrichedE2ECorpusRaw, EnrichedE2EItem,
                                                {'raw-input': lambda entry: extract_raw_input(entry),
                                                 'selected-input': extract_selected_input,
                                                 'ordered-input': extract_ordered_input,
                                                 'sentence-segmented-input': extract_sentence_segmented_input,
                                                 'lexicalisation': extract_lexicalization,
                                                 'referring-expressions': extract_reg_completed_lex_random,
                                                 'raw-output': extract_raw_output})

    corpus = EnrichedE2ECorpus(enriched_e2e_factory(corpus))
    logging.info(f"Corpus contains {len(corpus)} entries.")
    return corpus
