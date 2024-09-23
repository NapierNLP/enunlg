from collections import defaultdict
from typing import TYPE_CHECKING

import collections
import logging
import random

import torch

from enunlg.meaning_representation.slot_value import SlotValueMR, SlotValueMRList

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from enunlg.data_management.webnlg import RDFTriple, RDFTripleList

RegexRule = collections.namedtuple('RegexRule', ("match_expression", "replacement_expression"))


def count_parameters(model, log_table: bool = True, print_table: bool = False) -> int:
    """
    Based on https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model,
    forwarded to me by Jonas Groschwitz
    """
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if log_table:
        logger.info(table)
        logger.info(f"Total Trainable Params: {total_params}")
    if print_table:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params


def log_list_of_tensors_sizes(list_of_tensors, level=logging.DEBUG) -> None:
    logging.log(level, f"{len(list_of_tensors)=}")
    for task in list_of_tensors:
        logger.log(level, f"{task.size()}")


def log_sequence(seq, indent="") -> None:
    for element in seq:
        logger.info(f"{indent}{element}")


def set_random_seeds(seed) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def sv_to_rdf(mr) -> "RDFTripleList":
    from enunlg.data_management.webnlg import RDFTriple, RDFTripleList
    tripleset = []
    agent = mr['name']
    for slot in mr:
        if slot != "name":
            tripleset.append(RDFTriple(agent, slot, mr[slot]))
    return RDFTripleList(tripleset)


def hamming_error(target_bitvector, bitvector) -> float:
    return sum(abs(target_bitvector - bitvector))/sum(target_bitvector)


def translate_sv_corpus_to_rdf(corpus) -> None:
    for entry in corpus:
        agent = entry.raw_input['name']
        entry.raw_input = sv_to_rdf(entry.raw_input)
        entry.selected_input = sv_to_rdf(entry.selected_input)
        entry.ordered_input = sv_to_rdf(entry.ordered_input)
        sentence_mrs = []
        for sent_mr in entry.sentence_segmented_input:
            sent_mr_dict = dict(sent_mr)
            sent_mr_dict['name'] = agent
            sentence_mrs.append(sv_to_rdf(sent_mr_dict))
        entry.sentence_segmented_input = sentence_mrs


def translate_rdf_corpus_to_e2e(corpus) -> None:
    for entry in corpus:
        entry.raw_input = rdf_to_sv_list(entry.raw_input)
        entry.selected_input = rdf_to_sv_list(entry.selected_input)
        entry.ordered_input = rdf_to_sv_list(entry.ordered_input)
        sent_mr_lists = []
        for sent_rdf_list in entry.sentence_segmented_input:
            sent_mr_lists.append(rdf_to_sv_list(sent_rdf_list))
        entry.sentence_segmented_input = sent_mr_lists


def rdf_to_sv_set(rdf_triple_list) -> set:
    sv_set = set()
    for triple in rdf_triple_list:
        sv_set.add(('name', triple.subject))
        sv_set.add((triple.predicate, triple.object))
    return sv_set


def rdf_to_sv_list(rdf_triple_list) -> SlotValueMRList:
    grouped_by_name = defaultdict(list)
    relex_dict = {}
    for triple in rdf_triple_list:
        relex_dict.update(triple.relex_dict)
        grouped_by_name[triple.subject].append((triple.predicate, triple.object))
    mr_list = []
    for entity in grouped_by_name:
        mr = {'name': entity}
        for slot, value in grouped_by_name[entity]:
            mr[slot] = value
        mr_list.append(mr)
    mr_list = SlotValueMRList([SlotValueMR(mr) for mr in mr_list])
    mr_list.relex_dict = relex_dict
    return mr_list
