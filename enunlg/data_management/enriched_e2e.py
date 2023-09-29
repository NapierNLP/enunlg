from typing import Iterable, List, Optional, Union

import xsdata.formats.dataclass.parsers as xsparsers

from enunlg.formats.xml.enriched_e2e import Entries
import enunlg.data_management.iocorpus

# from enlg.data_management.pipelinecorpus import PipelineCorpus

# class EnrichedE2E(PipelineCorpus):
#     def __init__(self, seq: Optional[Iterable] = None, filename_or_list: Optional[Union[str, List[str]]] = None):
#         super().__init__(seq, filename_or_list, validation_schema = 'enlg/formats/enriched_e2e.xsd', entry_class: Optional[Callable] = None)


class EnrichedE2ECorpus(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[Iterable] = None, filename_or_list: Optional[Union[str, List[str]]] = None):
        if seq is None:
            seq = []
        super().__init__(seq)
        self.entries = []
        if filename_or_list is not None:
            if isinstance(filename_or_list, list):
                for filename in filename_or_list:
                    self.load_file(filename)
            elif isinstance(filename_or_list, str):
                self.load_file(filename_or_list)
            else:
                raise TypeError(f"Expected filename_or_list to be None, str, or list, not {type(filename_or_list)}")

    def load_file(self, filename):
        entries_object = xsparsers.XmlParser().parse(filename, Entries)
        self.entries.extend(entries_object.entry)


if __name__ == "__main__":
    tmp = EnrichedE2ECorpus()
    tmp.load_file('datasets/raw/EnrichedE2E/train/8attributes.xml')
    for x in tmp.entries[:10]:
        print(x)
