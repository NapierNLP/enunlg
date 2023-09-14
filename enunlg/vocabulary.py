from typing import Dict, Iterable, List, Union

import bidict


class IntegralRDFVocabulary(object):
    def __init__(self, dataset: Iterable[Iterable[object]]) -> None:
        """
        Embeddings for a set of dialogue acts such that each token receives a different integral value for each position it appears in.

        E.g. if 'name' occurs both as a slot and as a value, it will have different integers associated with it in each position.

        Not fully implemented because TGen is set up to only handle the 'inform' dialogue act, so we only actually use
        `IntegralInformEmbeddings` at the moment.
        :param dataset: an iterable over MRs
        """
        self._predicate_dict = bidict.OrderedBidict({'PRED_UNK': 0})
        # TODO consider using a version which uses different reps for subject and object versions of words
        self._arg_dict = bidict.OrderedBidict({'ARG_UNK': 1})
        self._filler = {0, 1}
        self._max_index = 1
        self._init_vocabulary(dataset)

    @property
    def max_index(self):
        return self._max_index

    @property
    def filler(self):
        return self._filler

    def _init_vocabulary(self, dataset) -> int:
        """
        Scan `self.dataset` and set up dicts to make it possible to generate embeddings.
        :return: the size of the vocabulary (i.e. the max index value associated with a dialogue act, slot, or value)
        """
        for rdf in dataset:
            for triple in rdf:
                self._add_predicate_if_new(triple.predicate)
                self._add_arg_if_new(triple.subject)
                self._add_arg_if_new(triple.object)
        return self._max_index

    def _add_predicate_if_new(self, predicate: str) -> None:
        if predicate not in self._predicate_dict:
            self._max_index += 1
            self._predicate_dict[predicate] = self._max_index

    def _add_arg_if_new(self, argument: str) -> None:
        if argument not in self._arg_dict:
            self._max_index += 1
            self._arg_dict[argument] = self._max_index

    def get_ints(self, rdf: Iterable[object]) -> List[int]:
        """
        Embed `rdf` based on the mappings built from `self.dataset`.

        Outputs will have length = 3 * len(rdf), with the mapping
        index % 3 = 0 --> integer representing the 'inform' dialogue act
        index % 3 = 1 --> integer represents a slot (assoc'd with value rep'd at index+1)
        index % 3 = 2 --> integer represents a value (assoc'd with slot rep'd at index-1)
        :param rdf: string-to-string dict representing an MR
        :return: a representation of the input MR as a sequence of integers
        """
        embedding = []
        for triple in rdf:
            embedding.append(self._predicate_dict[triple.predicate])
            embedding.append(self._arg_dict[triple.subject])
            embedding.append(self._arg_dict[triple.object])
        return embedding

    def get_ints_with_padding(self, rdf: Iterable[object], max_da_length: int = 10):
        embedding = self.get_ints(rdf)
        if len(embedding) > max_da_length * 3:
            # Truncate
            return embedding[:max_da_length * 3]
        else:
            # Left-pad
            padding = [self._predicate_dict['PRED_UNK'], self._arg_dict['ARG_UNK'], self._arg_dict['ARG_UNK']] * int((max_da_length * 3 - len(embedding)) / 3)
            return padding + embedding

    def get_tokens(self, rdf_integers: Iterable[int], drop_filler=True):
        output = []
        for index, integer in enumerate(rdf_integers):
            if drop_filler and integer in (0, 1):
                continue
            if index % 3 == 0:
                output.append(self._predicate_dict.inv[integer])
            else:
                output.append(self._arg_dict.inv[integer])
        return output

    def pretty_string(self, rdf_integers: Iterable[int], drop_filler=True) -> str:
        return " ".join(self.get_tokens(rdf_integers, drop_filler))


class IntegralDialogueActVocabulary(object):
    def __init__(self, dataset: Iterable[Dict[str, str]]) -> None:
        """
        Embeddings for a set of dialogue acts such that each token receives a different integral value for each position it appears in.

        E.g. if 'name' occurs both as a slot and as a value, it will have different integers associated with it in each position.

        Not fully implemented because TGen is set up to only handle the 'inform' dialogue act, so we only actually use
        `IntegralInformEmbeddings` at the moment.
        :param dataset: an iterable over MRs
        """
        self._act_dict = bidict.OrderedBidict({'ACT_UNK': 0})
        self._slot_dict = bidict.OrderedBidict({'SLOT_UNK': 1})
        self._value_dict = bidict.OrderedBidict({'VALUE_UNK': 2})
        self._filler = {0, 1, 2}
        self._max_index = 2

    @property
    def max_index(self):
        return self._max_index

    @property
    def filler(self):
        return self._filler

    def _init_vocabulary(self, dataset, multivalued_slots: bool) -> None:
        """
        Initialize the vocabulary based on self.dataset.

        :param multivalued_slots: whether the items in the dataset contain multiple values for each slot
        """
        # Note: if we choose to implement this, we need to check usages to make sure the call to __init__() for
        #       this class doesn't mess things up for subclasses after we implement it.
        raise NotImplementedError()


class IntegralInformVocabulary(IntegralDialogueActVocabulary):
    def __init__(self, dataset: Iterable[Dict[str, str]], multivalued_slots=False) -> None:
        """
        Embeddings for 'inform' dialogue acts with a collection of key-value pairs where each slot or value token
        receives a different integral value for each position it appears in.

        E.g. if 'name' occurs both as a slot and as a value, it will one integer associated with it in the slot position
        and another integer associated with it in the value position.

        Note that the current implementations are not designed for efficiency as processing the dataset in this way is
        assumed to not be a bottleneck in processing times.
        :param dataset: an iterable over MRs represented as dicts with string keys and values
        """
        super().__init__(dataset)
        self._max_index += 1
        self._act_dict['inform'] = self._max_index
        self._init_vocabulary(dataset, multivalued_slots)

    def _init_vocabulary(self, dataset, multivalued_slots: bool) -> int:
        """
        Scan `self.dataset` and set up dicts to make it possible to generate embeddings.
        :return: the size of the vocabulary (i.e. the max index value associated with a dialogue act, slot, or value)
        """
        for mr in dataset:
            for slot in mr:
                self._add_slot_if_new(slot)
                if multivalued_slots:
                    for value in mr[slot]:
                        self._add_value_if_new(value)
                else:
                    self._add_value_if_new(mr[slot])
        return self._max_index

    def _add_slot_if_new(self, slot: str) -> None:
        if slot not in self._slot_dict:
            self._max_index += 1
            self._slot_dict[slot] = self._max_index

    def _add_value_if_new(self, value: str) -> None:
        if value not in self._value_dict:
            self._max_index += 1
            self._value_dict[value] = self._max_index

    def get_ints(self, inform_da_dict: Dict[str, str]) -> List[int]:
        """
        Embed `inform_da_dict` based on the mappings built from `self.dataset`.

        Each slot-value pair is associated with the 'inform' dialogue act, so the output sequence will always be
        of length = 3 * len(inform_da_dict), with the mapping
        index % 3 = 0 --> integer representing the 'inform' dialogue act
        index % 3 = 1 --> integer represents a slot (assoc'd with value rep'd at index+1)
        index % 3 = 2 --> integer represents a value (assoc'd with slot rep'd at index-1)
        :param inform_da_dict: string-to-string dict representing an MR
        :return: a representation of the input MR as a sequence of integers
        """
        embedding = []
        for slot in inform_da_dict:
            embedding.append(self._act_dict['inform'])
            embedding.append(self._slot_dict[slot])
            embedding.append(self._value_dict[inform_da_dict[slot]])
        return embedding

    def get_ints_with_padding(self, inform_da_dict: Dict[str, str], max_da_length: int = 10):
        embedding = self.get_ints(inform_da_dict)
        if len(embedding) > max_da_length * 3:
            # Truncate
            return embedding[:max_da_length * 3]
        else:
            # Left-pad
            padding = [self._act_dict['ACT_UNK'], self._slot_dict['SLOT_UNK'], self._value_dict['VALUE_UNK']] * int((max_da_length * 3 - len(embedding)) / 3)
            return padding + embedding

    def get_tokens(self, da_integers: Iterable[int], drop_filler=True):
        output = []
        for index, integer in enumerate(da_integers):
            if drop_filler and integer in (0, 1, 2):
                continue
            if index % 3 == 0:
                output.append(self._act_dict.inv[integer])
            if index % 3 == 1:
                output.append(self._slot_dict.inv[integer])
            if index % 3 == 2:
                output.append(self._value_dict.inv[integer])
        return output

    def pretty_string(self, da_integers: Iterable[int], drop_filler=True) -> str:
        return " ".join(self.get_tokens(da_integers, drop_filler))


class TokenVocabulary(object):
    def __init__(self, dataset: Iterable[Iterable[str]]) -> None:
        """

        Note that the current implementations are not designed for efficiency as processing the dataset in this way is
        assumed to not be a bottleneck in processing times.
        :param dataset: an iterable over iterables of strings (i.e. pre-tokenised texts)
        """
        self.dataset = dataset
        self._token2int = bidict.OrderedBidict({
            '<VOID>': 0,
            '<GO>': 1,
            '<STOP>': 2,
            '<UNK>': 3,
            '<-s>': 4
        })
        self._max_index = 4
        self._filler = {0}
        self._init_vocabulary()

    @property
    def tokens(self):
        return list(self._token2int.keys())

    @property
    def filler(self):
        return self._filler

    @property
    def max_index(self):
        return self._max_index

    @property
    def stop_token_int(self):
        return self._token2int['<STOP>']

    def add_token(self, token) -> None:
        if token not in self._token2int:
            self._max_index += 1
            self._token2int[token] = self._max_index

    def _init_vocabulary(self) -> int:
        """
        Scan `self.dataset` and set up dicts to make it possible to generate embeddings.
        :return: the size of the vocabulary (i.e. the max index value associated with a dialogue act, slot, or value)
        """
        for sent in self.dataset:
            for token in sent:
                self.add_token(token)
        return self._max_index

    @staticmethod
    def _lowercase(token):
        """
        Lowercase a word token, keeping X-* placeholders + select all-caps words intact.

        copied directly from tgen.embeddings.TokenEmbeddingSeq2SeqExtract._lowercase()
        """
        if token is None or token in ['I', 'OK'] or token.startswith('X-'):
            return token
        return token.lower()

    def get_int(self, token: str) -> int:
        return self._token2int.get(token, self._token2int['<UNK>'])

    def get_ints(self, sentence: Iterable[str]) -> List[int]:
        """
        :param sentence: representation of a sentence as a sequence of TaggedToken tuples
        :return: a representation of the input sentence as a sequence of integers
        """
        embedding = [self._token2int['<GO>']]
        for token in sentence:
            embedding.append(self.get_int(token))
        embedding.append(self._token2int['<STOP>'])
        return embedding

    def get_ints_with_right_padding(self, sentence: Iterable[str], max_sentence_length: int = 50):
        """chore: add newline to end of file
        :param sentence: representation of a sentence as a sequence of TaggedToken tuples
        :param max_sentence_length: maximum allowed sentence length
        :return: a representation of the input sentence as a sequence of (`max_sentence_length` + 2) integers
        """
        embedding = self.get_ints(sentence)

        if len(embedding) > max_sentence_length + 2:
            # Truncate
            return embedding[:max_sentence_length + 2]
        elif len(embedding) < max_sentence_length + 2:
            # Right-pad
            padding = [self._token2int['<VOID>']] * (max_sentence_length + 2 - len(embedding))
            return embedding + padding
        else:
            return embedding

    def get_ints_with_left_padding(self, sentence: Iterable[str], max_sentence_length: int = 50):
        """
        :param sentence: representation of a sentence as a sequence of TaggedToken tuples
        :param max_sentence_length: maximum allowed sentence length
        :return: a representation of the input sentence as a sequence of (`max_sentence_length` + 2) integers
        """
        embedding = self.get_ints(sentence)

        if len(embedding) > max_sentence_length + 2:
            # Truncate
            return embedding[:max_sentence_length + 2]
        elif len(embedding) < max_sentence_length + 2:
            # Right-pad
            padding = [self._token2int['<VOID>']] * (max_sentence_length + 2 - len(embedding))
            return padding + embedding
        else:
            return embedding

    def get_token(self, token_integer: int) -> str:
        return self._token2int.inv[token_integer]

    def get_tokens(self, token_integers: Iterable[int], drop_filler=True) -> List[str]:
        """
        Map a sequence of integers to a sequence
        """
        output = []
        for integer in token_integers:
            if drop_filler and integer in self.filler:
                continue
            output.append(self.get_token(integer))
        return output

    def pretty_string(self, token_integers, drop_filler=True) -> str:
        return " ".join(self.get_tokens(token_integers, drop_filler))
