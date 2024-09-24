from collections import namedtuple
from typing import Iterable, List, Optional, Tuple

import csv
import logging
import os

from pyparsing import alphanums, Forward, OneOrMore, Suppress, Word, ZeroOrMore

import omegaconf

import enunlg.data_management.iocorpus as iocorpus

logger = logging.getLogger(__name__)

# TODO add hydra configuration for neural methodius stuff!
NEURAL_METHODIUS_DIR = os.path.join(os.path.dirname(__file__), '../../datasets/methodiusNeuralINLG2021/corpus/')
NEURAL_METHODIUS_CONFIG = omegaconf.DictConfig({'NEURAL_METHODIUS_DIR': NEURAL_METHODIUS_DIR})
NEURAL_METHODIUS_SPLITS = ('train_without_few', 'valid', 'test')

RSTNode = namedtuple("RSTNode", ["value", "children"])


def parse_action(string, location, tokens):
    return RSTNode(tokens[0], [RSTNode(x, []) if isinstance(x, str) else x for x in tokens[1:]])

neural_methodius_rst_grammar = Forward()
neural_methodius_rst_grammar << Suppress("[") + OneOrMore(Word(alphanums + "_.,-")) + ZeroOrMore(neural_methodius_rst_grammar) + Suppress("]")
neural_methodius_rst_grammar.set_parse_action(parse_action)


class MethodiusPair(iocorpus.IOPair):
    mr: RSTNode


class MethodiusCorpus(iocorpus.IOCorpus):
    def __init__(self, seq: Iterable[MethodiusPair]):
        super().__init__(seq)


def remove_tags(methodius_text_str: str) -> str:
    return methodius_text_str.replace("[text]", "").replace("[/text]", "").strip()


def parse_methodius_tree(methodius_tree_str: str) -> RSTNode:
    return neural_methodius_rst_grammar.parse_string(methodius_tree_str)[0]


def extract_facts(node):
    facts = []
    for child in node.children:
        if '__fact' in child.value:
            facts.append(child)
        else:
            facts.extend(extract_facts(child))
    return facts


def convert_fact_to_name_and_args(fact):
    name = fact.value.replace("__fact", "")
    args = []
    for child in fact.children:
        if child.value.startswith("__arg"):
            args.append(tuple(x.value for x in child.children))
        else:
            args.append( (child.value, ) )
    return name, args


def load_neural_methodius_tsv(filepath: str) -> List[Tuple[str, str]]:
    """E2E CSV files' first column is MRs and the second column is texts. There is always a header line."""
    with open(filepath, 'r') as in_file:
        tsv_reader = csv.reader(in_file, delimiter='\t')
        return MethodiusCorpus([MethodiusPair(parse_methodius_tree(pair[0]), remove_tags(pair[1])) for pair in tsv_reader])


def load_neural_methodius(splits: Optional[Iterable[str]] = None,
                          neural_methodius_config: Optional[omegaconf.DictConfig] = None) -> MethodiusCorpus:
    """

    :param splits: which splits to load
    :param original: True to load the original e2e corpus, false to load the cleaned version
    :param neural_methodius_config: a box.Box or omegaconf.DictConfig like object containing the basic
                       information about the e2e corpus to be used
    :return: the corpus of MR-text pairs with metadata
    """
    if neural_methodius_config is None:
        neural_methodius_config = NEURAL_METHODIUS_CONFIG
    corpus_name = "Neural Methodius Corpus"
    default_splits = NEURAL_METHODIUS_SPLITS
    directory = neural_methodius_config.NEURAL_METHODIUS_DIR
    if splits is None:
        splits = default_splits
    elif not set(splits).issubset(default_splits):
        message = f"`splits` can only contain a subset of {default_splits}. Found {splits}."
        raise ValueError(message)
    corpus = MethodiusCorpus([])
    for split in splits:
        corpus.extend(load_neural_methodius_tsv(os.path.join(directory, f"{split}.tsv")))
    corpus.metadata = {'name': corpus_name,
                       'splits': splits,
                       'directory': directory}
    return corpus


if __name__ == "__main__":
    data = load_neural_methodius()
    print(data[0].mr)
    print(extract_facts(data[0].mr))
    for fact in extract_facts(data[0].mr):
        print(convert_fact_to_name_and_args(fact))
    print(data[0].text)
