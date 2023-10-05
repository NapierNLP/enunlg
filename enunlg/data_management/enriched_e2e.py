from typing import Callable, Dict, Iterable, List, Optional, Union

import logging
import os

import box
import xsdata.formats.dataclass.parsers as xsparsers

from enunlg.formats.xml.enriched_e2e import EnrichedE2EEntries
import enunlg.data_management.iocorpus
import enunlg.data_management.pipelinecorpus

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

# TODO add hydra configuration for enriched e2e stuff
ENRICHED_E2E_CONFIG = box.Box({'ENRICHED_E2E_DIR': os.path.join(os.path.dirname(__file__), '../../datasets/raw/EnrichedE2E/')})

E2E_SPLIT_DIRS = ('train', 'dev', 'test')


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
        logging.info('successful init')

    def __call__(self, input_corpus):
        logging.info('successful call')
        if isinstance(input_corpus, self.input_format):
            logging.info('passed the format check')
            output_seq = []
            for x in input_corpus:
                output = []
                for layer in self.annotation_layer_mappings:
                    logging.info(f"processing {layer}")
                    output.append(self.annotation_layer_mappings[layer](x))
                max_length = max([len(x) for x in output])
                output = [x * max_length if len(x) == 1 else x for x in output]
                assert all([len(x) == max_length for x in output]), f"expected all layers to have the same number of items, but received: {[len(x) for x in output]}"
                output_seq.append(output)
                logging.info(len(output_seq))
            return self.output_format(output_seq)
        else:
            raise TypeError(f"Cannot run {self.__class__} on {type(input_corpus)}")


def load_enriched_e2e(splits: Optional[Iterable[str]] = None, enriched_e2e_config: Optional[box.Box] = None) -> EnrichedE2ECorpusRaw:
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
        logging.info(split)
        fns.extend([os.path.join(data_directory, split, fn) for fn in os.listdir(os.path.join(data_directory, split))])

    corpus = EnrichedE2ECorpusRaw(filename_or_list=fns)
    corpus.metadata = {'name': corpus_name,
                       'splits': splits,
                       'directory': data_directory}
    logging.info(len(corpus))
    #lambda entry: [x.sentence.content for x in entry.targets.structuring]

    def extract_ordered_input(entry):
        targets = []
        for target in entry.targets:
            selected_inputs = []
            for sentence in target.structuring.sentences:
                selected_inputs.extend(sentence.content)
            targets.append(selected_inputs)
        return targets

    def extract_sentence_segmented_input(entry):
        targets = []
        for target in entry.targets:
            selected_inputs = []
            for sentence in target.structuring.sentences:
                selected_inputs.append(sentence.content)
            targets.append(selected_inputs)
        return targets

    def extract_lexicalization(entry):
        return [target.lexicalization for target in entry.targets]

    def extract_raw_output(entry):
        return [target.text for target in entry.targets]

    enriched_e2e_factory = PipelineCorpusMapper(EnrichedE2ECorpusRaw, enunlg.data_management.iocorpus.IOCorpus,
                                                {'raw-input': lambda entry: [entry.source.inputs],
                                                 'selected-input': lambda entry: [set(x) for x in extract_ordered_input(entry)],
                                                 'ordered-input': extract_ordered_input,
                                                 'sentence-segmented-input': extract_sentence_segmented_input,
                                                 'lexicalisation': extract_lexicalization,
                                                 'referring-expressions': lambda x: [None],
                                                 'raw-output': extract_raw_output})
    logging.info(len(corpus))
    corpus = enriched_e2e_factory(corpus)
    logging.info(len(corpus))
    return corpus
