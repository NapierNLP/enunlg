from typing import Callable, Iterable, List, Optional, Union

import lxml.etree

import enunlg.data_management.iocorpus


class PipelineCorpus(enunlg.data_management.iocorpus.IOCorpus):
    def __init__(self, seq: Optional[Iterable] = None, filename_or_list: Optional[Union[str, List[str]]] = None, validation_schema: Optional[str] = None, entry_class: Optional[Callable] = None):
        if seq is None:
            seq = []
        super().__init__(seq)
        self.entries = []
        if validation_schema is not None:
            self.validation_schema_doc = lxml.etree.parse(validation_schema)
        if entry_class is not None:
            self.entry_class = entry_class
        if filename_or_list is not None:
            if isinstance(filename_or_list, list):
                for filename in filename_or_list:
                    self.load_file(filename)
            elif isinstance(filename_or_list, str):
                    with open(filename_or_list, 'rb') as fs:
                        self.validate(fs)
                        self.add_entries_from_file(fs)
            else:
                raise TypeError(f"Expected filename_or_list to be None, str, or list, not {type(filename_or_list)}")

    def load_file(self, filename):
        with open(filename, 'rb') as fs:
            if self.validate(fs):
                self.add_entries_from_file(fs)
            else:
                raise ValueError(f"Invalid xml file: {filename}")

    def validate(self, filestream):
        self.validation_schema_doc.validate(filestream)

    def add_entries_from_file(self, filestream) -> None:
        entries_xml = lxml.etree.parse(filestream).getroot()
        entries_xml = entries_xml.findall(".//entry")
        for entry_xml in entries_xml:
            self.entries.append(self.entry_class(entry_xml))
