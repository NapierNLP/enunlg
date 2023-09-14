from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable


@dataclass
class IOPair:
    __slots__ = ("mr", "text")
    mr: Any
    text: str

    def __iter__(self):
        return iter((self.mr, self.text))


class Corpus(list):
    def __init__(self, seq: Iterable) -> None:
        """
        A corpus is a list of items with associated metadata
        """
        super().__init__(seq)

        self.metadata: Optional[Dict[str, Any]] = None


class IOCorpus(Corpus):
    def __init__(self, seq: Iterable) -> None:
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
