from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable

import logging

logger = logging.getLogger(__name__)


@dataclass
class IOPair:
    __slots__ = ("mr", "text")
    mr: Any
    text: str

    def __getitem__(self, item):
        return self.__getattribute__(self.__slots__[item])

    def __iter__(self):
        return iter((self.mr, self.text))


class Corpus(list):
    def __init__(self, seq: Optional[Iterable]) -> None:
        """
        A corpus is a list of items with associated metadata
        """
        if seq is None:
            seq = []
        super().__init__(seq)

        self.metadata: Optional[Dict[str, Any]] = None

    def __getitem__(self, key):
        if isinstance(key, slice):
            retval = self.__class__(super().__getitem__(key))
            retval.metadata = self.metadata
            return retval
        return super().__getitem__(key)


class IOCorpus(Corpus):
    def __init__(self, seq: Optional[Iterable]) -> None:
        """
        An IOCorpus is a corpus of input-output pairs.
        :param seq:
        """
        super().__init__(seq)

    @property
    def inputs(self) -> List:
        return [i for i, _ in self]

    @property
    def outputs(self) -> List:
        return [o for _, o in self]
