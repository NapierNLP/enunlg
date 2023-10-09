"""Lookup templates use MRs as keys and texts as values."""

from collections import defaultdict
from typing import Any, Hashable, Iterable, Tuple

import abc
import random


class LookupGenerator(abc.ABC):
    @abc.abstractmethod
    def _add_io_pair(self, pair) -> None:
        pass

    def train(self, pairs: Iterable[Tuple[Hashable, Any]]) -> None:
        for pair in pairs:
            self._add_io_pair(pair)

    @abc.abstractmethod
    def predict(self, mr):
        pass


class OneToOneLookupGenerator(LookupGenerator):
    def __init__(self) -> None:
        """
        LookupGenerator which maintains a single template text for each MR.

        NB: There is no method for choosing which template gets stored for a given MR;
        any time a new (MR, Text) pair is seen in training, it replaces the previous Text for that MR.
        """
        self._mapping = {}

    def _add_io_pair(self, io_pair: Tuple[Hashable, Any]) -> None:
        self._mapping[io_pair[0]] = io_pair[1]

    def train(self, pairs: Iterable[Tuple[Hashable, Any]]) -> None:
        for pair in pairs:
            self._add_io_pair(pair)

    def predict(self, mr: Hashable) -> str:
        return self._mapping[mr]


class OneToManyLookupGenerator(LookupGenerator):
    def __init__(self) -> None:
        """
        LookupGenerator which keeps a list of templates for each MR.

        NB: Templates are chosen randomly at prediction time. Since template texts are stored in a list,
        the random choice at prediction time will be weighted by how frequently each text was seen during training.
        """
        self._mapping = defaultdict(list)

    def _add_io_pair(self, io_pair: Tuple[Hashable, Any]) -> None:
        self._mapping[io_pair[0]].append(io_pair[1])

    def predict(self, mr: Hashable) -> Any:
        return random.choice(self._mapping[mr])
