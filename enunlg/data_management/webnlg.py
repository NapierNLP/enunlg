"""Code for working with WebNLG datasets as described at https://synalp.gitlabpages.inria.fr/webnlg-challenge/docs/"""

from typing import List, Optional, Union

import logging
import os

from lxml import etree

import enunlg.data_management.iocorpus

logger = logging.getLogger(__name__)

WEBNLG_2023_DIR = os.path.join(os.path.dirname(__file__), '../../datasets/raw/2023-Challenge/data/')


class RDFTriple(object):
    # it would be nice to use a dataclass for this, but then we can't call the object attribute object easily
    def __init__(self, subj, pred, obj):
        self.subject = subj
        self.predicate = pred
        self.object = obj
        self.relex_dict = {}

    def __repr__(self):
        return f"RDFTriple({self.subject}, {self.predicate}, {self.object})"

    def __hash__(self):
        return self.__repr__().__hash__()

    @staticmethod
    def from_string(triple: str) -> "RDFTriple":
        subj, pred, obj = triple.split(" | ")
        return RDFTriple(subj.strip('"'), pred.strip('"'), obj.strip('"'))

    def delex_reference(self, entity, sem_class):
        if entity == self.subject:
            self.relex_dict[sem_class] = entity
            self.subject = self.subject.replace(entity, sem_class)
        if entity == self.object:
            self.relex_dict[sem_class] = entity
            self.object = self.object.replace(entity, sem_class)

    def can_delex(self, entity: str) -> bool:
        return entity == self.subject or entity == self.object


class RDFTripleSet(set):
    def __init__(self, seq):
        super().__init__(seq)


class RDFTripleList(list):
    def __init__(self, seq):
        super().__init__(seq)

    def delex_reference(self, entity, sem_class):
        if self.can_delex(entity):
            for triple in self:
                triple.delex_reference(entity, sem_class)

    def can_delex(self, entity: str) -> bool:
        return any([triple.can_delex(entity) for triple in self])

    def find_pred_for_object(self, entity: str) -> Optional[str]:
        for triple in self:
            if triple.object == entity:
                return triple.predicate
        # print(entity)
        # print(self)


class WebNLGLex(object):
    def __init__(self, text=None, lex_id=None, lang=None, comment=None):
        self.text = text
        self.lex_id = lex_id
        self.lang = lang
        self.comment = comment

    @staticmethod
    def from_xml(lex_xml):
        return WebNLGLex(lex_xml.text,
                         lex_xml.attrib.get('lid'),
                         lex_xml.attrib.get('lang'),
                         lex_xml.attrib.get('comment'))


class RDFPair(enunlg.data_management.iocorpus.IOPair):
    mr: Union[RDFTripleSet, RDFTripleList]

    @property
    def rdf(self):
        return self.mr

    @rdf.setter
    def rdf(self, value):
        self.mr = value

    def sort_mr(self, in_place: bool = True) -> Optional[RDFTripleList]:
        # We'll use a list instead of a set for the ordered version
        sorted_mr = RDFTripleList(sorted(self.mr, key=lambda x: (x.predicate, x.subject, x.object)))
        if in_place:
            self.mr = sorted_mr
        else:
            return sorted_mr
        # Added for PEP8 consistency and mypy happiness
        return None

    def sort_rdf(self, in_place: bool = True) -> Optional[RDFTripleList]:
        return self.sort_mr(in_place)


class WebNLGEntry(object):
    def __init__(self, xml_entry=None):
        self.xml_entry = xml_entry
        self.dbpedia_category = None
        self.entry_id = None
        self.shape = None
        self.shape_type = None

        self.original_tripleset = set()
        self.modified_tripleset = set()
        self.texts = []
        if xml_entry is not None:
            self.from_xml_entry(xml_entry)

    @property
    def num_triples(self):
        return len(self.original_tripleset)

    def from_xml_entry(self, xml_entry):
        # TODO add check for unexpected attributes, in case someone adds a comment or property or sthg in the future
        self.dbpedia_category = xml_entry.attrib.get('category')
        self.entry_id = xml_entry.attrib.get('eid')
        self.shape = xml_entry.attrib.get('shape')
        self.shape_type = xml_entry.attrib.get('shape_type')
        reported_size = int(xml_entry.attrib.get('size'))
        for child in xml_entry:
            if child.tag == 'originaltripleset':
                triples = [RDFTriple.from_string(triple.text) for triple in child]
                if len(triples) != reported_size:
                    message = f"Size mismatch. <entry> reports size {reported_size} but found {len(triples)} triples."
                    raise ValueError(message)
                self.original_tripleset = RDFTripleSet(triples)
            elif child.tag == 'modifiedtripleset':
                triples = [RDFTriple.from_string(triple.text) for triple in child]
                if len(triples) != reported_size:
                    message = f"Size mismatch. <entry> reports size {reported_size} but found {len(triples)} triples."
                    raise ValueError(message)
                self.modified_tripleset = RDFTripleSet(triples)
            elif child.tag == 'lex':
                self.texts.append(WebNLGLex.from_xml(child))
            else:
                message = f"Unexpected child of the XML <entry> with tag: <{child.tag}>"
                raise ValueError(message)


class WebNLGCorpus(object):
    def __init__(self, filename: Optional[str] = None) -> None:
        self.entries: List[WebNLGEntry] = []
        if filename is not None:
            self.add_entries_from_file(filename)

    def add_entries_from_file(self, filename: str) -> None:
        with open(filename, 'rb') as webnlg_file:
            webnlg_xml = etree.parse(webnlg_file).getroot()
            entries_xml = webnlg_xml.findall(".//entry")
            for entry_xml in entries_xml:
                self.entries.append(WebNLGEntry(entry_xml))


class RDFTextCorpus(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq):
        super(RDFTextCorpus, self).__init__(seq)


def select_webnlg_pairs_by_language(webnlg_corpus: WebNLGCorpus, language_code: str, prefer_modified_triples=True) -> RDFTextCorpus:
    pairs = []
    for entry in webnlg_corpus.entries:
        for lex in entry.texts:
            if lex.lang == language_code:
                if entry.modified_tripleset:
                    rdf = entry.modified_tripleset
                else:
                    rdf = entry.original_tripleset
                pairs.append(RDFPair(mr=rdf, text=lex.text))
    return RDFTextCorpus(pairs)


if __name__ == "__main__":
    webnlg2023 = WebNLGCorpus(os.path.join(WEBNLG_2023_DIR, 'br_dev.xml'))
    pairs = select_webnlg_pairs_by_language(webnlg2023, 'br')
