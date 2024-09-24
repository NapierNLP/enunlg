from collections import defaultdict
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Iterable, List, Optional, Tuple, Union

import difflib
import logging
import os

import omegaconf
import regex
import xsdata.formats.dataclass.parsers.handlers.lxml
import xsdata.formats.dataclass.parsers as xsparsers

from enunlg.formats.xml.enriched_e2e import EnrichedE2EEntries, EnrichedE2EEntry
from enunlg.meaning_representation.slot_value import SlotValueMR, SlotValueMRList
from enunlg.normalisation.tokenisation import TGenTokeniser

import enunlg.data_management.pipelinecorpus

logger = logging.getLogger(__name__)

# TODO add hydra configuration for enriched e2e stuff
ENRICHED_E2E_CONFIG = omegaconf.DictConfig({'ENRICHED_E2E_DIR': os.path.join(os.path.dirname(__file__),
                                                                             '../../datasets/processed/EnrichedE2E/')})

E2E_SPLIT_DIRS = ('train', 'dev', 'test')

DELEX_LABELS = ["__AREA__", "__CUSTOMER_RATING__", "__EATTYPE__", "__FAMILYFRIENDLY__", "__FOOD__", "__NAME__",
                "__NEAR__", "__PRICERANGE__"]

DIFFER = difflib.Differ()


def extract_reg_from_template_and_text(text: str, template: str, print_diff: bool = False) -> MutableMapping[str, List[str]]:
    diff = DIFFER.compare(text.strip().split(), template.strip().split())
    keys = []
    values = []
    curr_key = []
    curr_value = []
    for line in diff:
        if print_diff:
            logger.debug(line)
        if line.startswith('-'):
            curr_value.append(line.split()[1])
        elif line.startswith('+'):
            token = line.split()[1]
            curr_key.append(token)

        else:
            if curr_value:
                values.append(" ".join(curr_value))
                curr_value = []
            if curr_key:
                keys.append(" ".join(curr_key))
                curr_key = []
    # Add the last key & value to the dict if we have some
    if curr_value:
        values.append(" ".join(curr_value))
    if curr_key:
        keys.append(" ".join(curr_key))
    result_dict = defaultdict(list)
    for key, value in zip(keys, values):
        result_dict[key].append(value)
    return result_dict


class EnrichedE2ECorpusRaw(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[Iterable] = None, filename_or_list: Optional[Union[str, List[str]]] = None):
        super().__init__(seq)
        if filename_or_list is not None:
            if isinstance(filename_or_list, list):
                for filename in filename_or_list:
                    logger.info(filename)
                    self.load_file(filename)
            elif isinstance(filename_or_list, str):
                self.load_file(filename_or_list)
            else:
                message = f"Expected filename_or_list to be None, str, or list, not {type(filename_or_list)}"
                raise TypeError(message)

    def load_file(self, filename):
        entries_object = xsparsers.XmlParser(handler=xsdata.formats.dataclass.parsers.handlers.lxml.LxmlEventHandler).parse(filename, EnrichedE2EEntries)
        # entries_object = xsparsers.XmlParser().parse(filename, EnrichedE2EEntries)
        self.extend(entries_object.entries)


class EnrichedE2EItem(enunlg.data_management.pipelinecorpus.PipelineItem):
    def __init__(self, annotation_layers):
        super().__init__(annotation_layers)

    def delex_slots(self, slots):
        for slot in slots:
            if slot not in self['raw_input']:
                continue
            value = self['raw_input'][slot]
            # print(value)
            for layer_name in self.annotation_layers:
                layer = self[layer_name]
                orig_tag = f"__{slot.upper()}__"
                if isinstance(layer, SlotValueMR):
                    layer.delex_slot(slot)
                elif isinstance(layer, tuple):
                    if all(isinstance(element, SlotValueMR) for element in layer):
                        for element in layer:
                            element.delex_slot(slot)
                elif isinstance(layer, str):
                    if layer_name == 'raw_output':
                        new_output = layer.replace(value, orig_tag).replace(value.replace("_", " "), orig_tag)
                        if new_output == layer:
                            logger.debug(f"Could not find '{value}' in:\n\t{layer}")
                        self[layer_name] = new_output
                else:
                    raise ValueError(f"Unexpected type for this layer: {type(layer)}")
        for slot in self['raw_input']:
            if slot not in slots:
                self['lexicalisation'] = self['lexicalisation'].replace(f"__{slot.replace(" ", "_").upper()}__", self['raw_input'][slot])

    def can_delex(self, slots):
        can_delex = True
        for slot in slots:
            if slot not in self['raw_input']:
                continue
            value = self['raw_input'][slot]
            for layer_name in self.annotation_layers:
                layer = self[layer_name]
                orig_tag = f"__{slot.upper()}__"
                if isinstance(layer, SlotValueMR):
                    if not layer.can_delex(slot):
                        can_delex = False
                elif isinstance(layer, tuple):
                    if all(isinstance(element, SlotValueMR) for element in layer):
                        if not any(element.can_delex(slot) for element in layer):
                            can_delex = False
                elif isinstance(layer, str):
                    if layer_name == 'lexicalisation':
                        if orig_tag not in self[layer_name]:
                            can_delex = False
                    else:
                        if value.replace("_", " ") not in self[layer_name] and value not in self[layer_name]:
                            can_delex = False
                else:
                    raise ValueError(f"Unexpected type for this layer: {type(layer)}")
        return can_delex


class EnrichedE2ECorpus(enunlg.data_management.pipelinecorpus.PipelineCorpus):
    def __init__(self, seq: List[EnrichedE2EItem], metadata=None):
        super(EnrichedE2ECorpus, self).__init__(seq, metadata)

    def validate_enriched_e2e(self) -> None:
        entries_to_drop = []
        for idx, entry in enumerate(self):
            # Some of the EnrichedE2E entries have incorrect semantics.
            # Checking for the restaurant name in the input selections is the fastest way to check.
            if 'name' in entry.raw_input and 'name' in entry.selected_input and 'name' in entry.ordered_input and entry.raw_output.strip() and entry.lexicalisation.strip():
                pass
            else:
                entries_to_drop.append(idx)
        for idx in reversed(entries_to_drop):
            self.pop(idx)

    def delexicalise_by_slot_name(self, slots):
        undelexicalisable_entries = []
        for idx, entry in enumerate(self):
            # logger.debug("-=-=-=-=-=-=-=-==-")
            # logger.debug(f"Attempting to delex entry #{idx}")
            # logger.debug(entry)
            orig_entry = deepcopy(entry)
            if entry.can_delex(slots):
                entry.delex_slots(slots)
            if repr(orig_entry) == repr(entry):
                logger.debug(f"Could not delex: {entry}")
                undelexicalisable_entries.append(idx)
            # if any(f"__{x.upper()}__" in entry['lexicalisation'] for x in slots):
            #     logger.info(f"the lex layer is not fully delexicalised: {entry['lexicalisation']}")
            #     logger.debug(f"could not fully delexicalise the lexicalisation layer for {entry}")
        logger.info(f"We had to discard {len(undelexicalisable_entries)} entries as undelexicalisable.")
        for idx in reversed(undelexicalisable_entries):
            self.pop(idx)

    @classmethod
    def from_raw_corpus(cls, corpus):
        out_corpus = []
        for entry in corpus:
            raw_input = extract_raw_input(entry)
            for target in entry.targets:
                selected_mr = {}
                sentence_grouped_mrs = []
                for sentence in target.structuring.sentences:
                    sent_mr = {}
                    for input_element in sentence.content:
                        selected_mr[input_element.attribute] = input_element.value
                        sent_mr[input_element.attribute] = input_element.value
                    sentence_grouped_mrs.append(SlotValueMR(sent_mr))  # , frozen_box=True))
                selected_input = SlotValueMR(selected_mr)
                ordered_input = SlotValueMR(deepcopy(selected_mr))
                sentence_segmented_input = tuple(sentence_grouped_mrs)
                lexicalisation = target.lexicalization.replace(" @ ", " ")
                raw_output = target.text.replace(" @ ", " ")
                # This will drop any entries which contain 'None' for any annotation layers
                if None in [raw_output, target.template, lexicalisation]:
                    continue
                if any([None in x for x in [raw_input, selected_input, ordered_input, sentence_segmented_input]]):
                    continue
                if any([len(x) == 0 for x in [raw_input, selected_input, ordered_input, sentence_segmented_input]]):
                    continue
                new_item = EnrichedE2EItem({'raw_input': deepcopy(raw_input),
                                            'selected_input': selected_input,
                                            'ordered_input': ordered_input,
                                            'sentence_segmented_input': sentence_segmented_input,
                                            'lexicalisation': lexicalisation,
                                            'raw_output': raw_output})
                new_item.metadata['eid'] = entry.eid
                new_item.metadata['lid'] = target.lid
                out_corpus.append(new_item)
        return cls(out_corpus)


def extract_raw_input(entry: EnrichedE2EEntry) -> SlotValueMR:
    mr = {}
    for source_input in entry.source.inputs:
        mr[source_input.attribute] = source_input.value
    return SlotValueMR(mr)  #, frozen_box=True)]


def extract_reg_in_lex(entry: EnrichedE2EEntry) -> List[str]:
    texts = [target.text for target in entry.targets]
    templates = [target.template for target in entry.targets]
    lexes = [target.lexicalization for target in entry.targets]
    reg_lexes = []
    for text, template, lex in zip(texts, templates, lexes):
        reg_dict = extract_reg_from_template_and_text(text, template)
        new_lex = []
        curr_text_idx = 0
        for lex_token in lex.split():
            if lex_token.startswith("__"):
                # print(f"looking for {lex_token}")
                possible_targets = reg_dict[lex_token]
                match_found = False
                curr_rest = text.split()[curr_text_idx:]
                for possible_target in possible_targets:
                    target_tokens = tuple(possible_target.split())
                    num_tokens = len(target_tokens)
                    # print(target_tokens)
                    # print(curr_rest)
                    if num_tokens == 1:
                        for text_idx, text_token in enumerate(curr_rest):
                            if text_token in possible_targets:
                                new_lex.append(text_token)
                                curr_text_idx += text_idx
                                match_found = True
                                break
                    elif num_tokens > 1:
                        parts = [curr_rest[i:] for i in range(len(target_tokens))]
                        for start_idx, token_tuple in enumerate(zip(*parts)):
                            # print(token_tuple)
                            if token_tuple == target_tokens:
                                new_lex.extend(target_tokens)
                                curr_text_idx += start_idx + len(target_tokens)
                                match_found = True
                                break
                    else:
                        message = "Must have possible targets for each slot!"
                        raise ValueError(message)
                    if match_found:
                        break
                if not match_found:
                    logger.debug(f"Could not create reg_lex text for {lex_token}")
                    logger.debug(f"in:\n{text}\n{template}\n{lex}\n{reg_dict}")
                    extract_reg_from_template_and_text(text, template, print_diff=True)
                    new_lex.append(lex_token)
                    # raise ValueError(f"Could not create reg_lex text for {lex_token} in:\n{text}\n{template}\n{lex}\n{reg_dict}")
            else:
                new_lex.append(lex_token)
        reg_lexes.append(" ".join(new_lex))
    return reg_lexes


def load_enriched_e2e(enriched_e2e_config: omegaconf.DictConfig, splits: Optional[Iterable[str]] = None) -> EnrichedE2ECorpus:
    """
    :param enriched_e2e_config: omegaconf.DictConfig like object containing the basic
                                information about the e2e corpus to be used
    :param splits: which splits to load
    :return: the corpus of MR-text pairs with metadata
    """
    default_splits = set(enriched_e2e_config.splits.keys())
    if not set(splits).issubset(default_splits):
        message = f"`splits` can only contain a subset of {default_splits}. Found {splits}."
        raise ValueError(message)
    fns = []
    for split in splits:
        logger.info(split)
        fns.extend([os.path.join(os.path.dirname(__file__), enriched_e2e_config.load_dir, split, fn)
                    for fn in os.listdir(os.path.join(os.path.dirname(__file__), enriched_e2e_config.load_dir, split))])

    corpus: EnrichedE2ECorpusRaw = EnrichedE2ECorpusRaw(filename_or_list=fns)
    corpus.metadata = {'name': enriched_e2e_config.display_name,
                       'splits': splits,
                       'directory': enriched_e2e_config.load_dir,
                       'raw': True}
    logger.info(f"Corpus size: {len(corpus)}")

    # tokenize texts
    for entry in corpus:
        for target in entry.targets:
            target.text = TGenTokeniser.tokenise(target.text)

    # Specify the type again since we're changing the expected type of the variable and mypy doesn't like that
    corpus: EnrichedE2ECorpus = EnrichedE2ECorpus.from_raw_corpus(corpus)
    corpus.metadata = {'name': enriched_e2e_config.display_name,
                       'splits': splits,
                       'directory': enriched_e2e_config.load_dir,
                       'raw': False}
    logger.info(f"Corpus contains {len(corpus)} entries.")
    return corpus


def sanitize_values(value):
    return value.replace(" ", "_").replace("'", "_")


snake_case_regex = regex.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')


def tokenize_slots_and_values(value):
    out_string = snake_case_regex.sub(r'_\1', value)
    tokens = []
    for token in out_string.split(" "):
        if token.startswith("__") and token.endswith("__"):
            tokens.append(token)
        else:
            tokens.append(token.replace("_", " ").replace(",", " , "))
    out_string = " ".join(tokens)
    out_tokens = out_string.split()
    # omit empty strings caused by multiple spaces in a row
    return [token for token in out_tokens if token]


def sanitize_slot_names(slot_name):
    return slot_name


def linearize_slot_value_mr(mr: enunlg.meaning_representation.slot_value.SlotValueMR):
    tokens = ["<MR>"]
    for slot in mr:
        tokens.extend([x.lower() for x in tokenize_slots_and_values(slot)])
        tokens.append("==")
        tokens.extend(tokenize_slots_and_values(mr[slot]))
        tokens.append("<PAIR_SEP>")
    tokens.append("</MR>")
    return tokens


def linearize_slot_value_mr_simplified(mr: enunlg.meaning_representation.slot_value.SlotValueMR):
    tokens = []
    for slot in mr:
        tokens.extend([x.lower() for x in tokenize_slots_and_values(slot)])
        tokens.append("==")
        tokens.extend(tokenize_slots_and_values(mr[slot]))
        tokens.append("<PAIR_SEP>")
    return tokens[:-1]


def linearize_slot_value_mr_seq(mrs, tag_label="SENTENCE"):
    tokens = []
    for mr in mrs:
        tokens.append(f"<{tag_label}>")
        tokens.extend(linearize_slot_value_mr(mr))
        tokens.append(f"</{tag_label}>")
    return tokens


def linearize_slot_value_mr_seq_simplified(mrs, tag_label="SENTENCE"):
    tokens = []
    for mr in mrs:
        tokens.extend(linearize_slot_value_mr_simplified(mr))
        tokens.append(f"<{tag_label}_SEP>")
    return tokens[:-1]


ORIG_LINEARIZATION_FUNCTIONS = {'raw_input': linearize_slot_value_mr,
                           'selected_input': linearize_slot_value_mr,
                           'ordered_input': linearize_slot_value_mr,
                           'sentence_segmented_input': linearize_slot_value_mr_seq,
                           'lexicalisation': lambda lex_string: lex_string.strip().replace(" @ ", " ").split(),
                           'referring_expressions': lambda reg_string: reg_string.strip().replace(" @ ", " ").split(),
                           'raw_output': lambda text: text.strip().split()}


LINEARIZATION_FUNCTIONS = {'raw_input': linearize_slot_value_mr_simplified,
                           'selected_input': linearize_slot_value_mr_simplified,
                           'ordered_input': linearize_slot_value_mr_simplified,
                           'sentence_segmented_input': linearize_slot_value_mr_seq_simplified,
                           'lexicalisation': lambda lex_string: lex_string.strip().replace(" @ ", " ").split(),
                           'referring_expressions': lambda reg_string: reg_string.strip().replace(" @ ", " ").split(),
                           'raw_output': lambda text: text.strip().split()}


def linearize_slot_value_mr_list(mr_list: SlotValueMRList):
    return linearize_slot_value_mr_seq_simplified(mr_list, "MR_LIST")


def wrap_in_sentence_tags(list_of_mr_lists):
    seq = []
    for mr_list in list_of_mr_lists:
        seq.append("<SENTENCE>")
        seq.extend(linearize_slot_value_mr_list(mr_list))
        seq.append("</SENTENCE>")
    return seq


def linearize_slot_value_mr_list_simplified(mr_list: SlotValueMRList):
    return linearize_slot_value_mr_seq_simplified(mr_list, "MR_LIST")


def wrap_in_sentence_tags_simplified(list_of_mr_lists):
    seq = []
    for mr_list in list_of_mr_lists:
        seq.extend(linearize_slot_value_mr_list(mr_list))
        seq.append("<SENTENCE_SEP>")
    return seq[:-1]


ORIG_LINEARIZATION_FUNCTIONS_WITH_SLOTVALUE_LISTS = {'raw_input': linearize_slot_value_mr_list,
                                                'selected_input': linearize_slot_value_mr_list,
                                                'ordered_input': linearize_slot_value_mr_list,
                                                'sentence_segmented_input': wrap_in_sentence_tags,
                                                'lexicalisation': lambda lex_string: lex_string.strip().replace(" @ ", " ").split(),
                                                'referring_expressions': lambda reg_string: reg_string.strip().replace(" @ ", " ").split(),
                                                'raw_output': lambda text: text.strip().split()}


LINEARIZATION_FUNCTIONS_WITH_SLOTVALUE_LISTS = {'raw_input': linearize_slot_value_mr_list_simplified,
                                                'selected_input': linearize_slot_value_mr_list_simplified,
                                                'ordered_input': linearize_slot_value_mr_list_simplified,
                                                'sentence_segmented_input': wrap_in_sentence_tags_simplified,
                                                'lexicalisation': lambda lex_string: lex_string.strip().replace(" @ ", " ").split(),
                                                'referring_expressions': lambda reg_string: reg_string.strip().replace(" @ ", " ").split(),
                                                'raw_output': lambda text: text.strip().split()}
