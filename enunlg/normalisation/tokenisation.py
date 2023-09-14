"""
Classes, functions, and constants for tokenisation.
"""

from collections import namedtuple
from typing import List

import abc

import regex

RegexRule = namedtuple(typename='RegexRule',
                       field_names=("match_expression", "replacement_expression"))


class AbstractTokeniser(abc.ABC):
    @classmethod
    def preprocess(cls, text: str):
        return text

    @classmethod
    def postprocess(cls, toks) -> str:
        return toks

    @classmethod
    @abc.abstractmethod
    def tokenise(cls, text: str) -> str:
        pass

    @classmethod
    def tokenize(cls, text: str) -> str:
        return cls.tokenise(text)

    @classmethod
    def tokenise_to_list(cls, text: str) -> List[str]:
        return cls.tokenise(text).split()

    @classmethod
    def tokenize_to_list(cls, text: str) -> List[str]:
        return cls.tokenise_to_list(text)


class ExistingWhitespaceTokeniser(AbstractTokeniser):
    @classmethod
    def tokenise(cls, text: str) -> str:
        return text


class TGenTokeniser(AbstractTokeniser):
    # Three sets of rules. First set enforces spaces around punctuation.
    rules = (RegexRule(r'(([^\p{IsAlnum}\s\.\,−\-])\2*)', r' \1 '),
             RegexRule(r'([^\p{N}])([,.])([^\p{N}])', r'\1 \2 \3'),
             RegexRule(r'([^\p{N}])([,.])([\p{N}])', r'\1 \2 \3'),
             RegexRule(r'([\p{N}])([,.])([^\p{N}])', r'\1 \2 \3'),
             RegexRule(r'(–-)([^\p{N}])', r'\1 \2'),
             RegexRule(r'(\p{N} *|[^ ])(-)', r'\1\2 '),
             RegexRule(r'([-−])', r' \1'),
             # Second set keeps apostrophes together with words in most common contractions.
             RegexRule(r'([\'’´]) (s|m|d|ll|re|ve)\s', r' \1\2 '),
             RegexRule(r'(n [\'’´]) (t\s)', r' \1\2 '),
             # Third set of contractions based on Treex.
             RegexRule(r' ([Cc])annot\s', r' \1an not '),
             RegexRule(r' ([Dd]) \' ye\s', r' \1\' ye '),
             RegexRule(r' ([Gg])imme\s', r' \1im me '),
             RegexRule(r' ([Gg])onna\s', r' \1on na '),
             RegexRule(r' ([Gg])otta\s', r' \1ot ta '),
             RegexRule(r' ([Ll])emme\s', r' \1em me '),
             RegexRule(r' ([Mm])ore\'n\s', r' \1ore \'n '),
             RegexRule(r' \' ([Tt])is\s', r' \'\1 is '),
             RegexRule(r' \' ([Tt])was\s', r' \'\1 was '),
             RegexRule(r' ([Ww])anna\s', r' \1an na ')
             )

    def __init__(self):
        """Class providing TGen's tokenisation rules for English texts."""
        pass

    @classmethod
    def preprocess(cls, text: str):
        """TGen inserts spaces around text for easier regexes"""
        return f" {text} "

    @classmethod
    def postprocess(cls, toks: str):
        """TGen removes extra spaces from the text, so spaces can be used as token separators."""
        return regex.sub(r'\s+', ' ', toks).strip()

    @classmethod
    def tokenise(cls, text: str) -> str:
        intermediate_text = cls.preprocess(text)
        for rule in cls.rules:
            intermediate_text = regex.sub(rule.match_expression, rule.replacement_expression, intermediate_text)
        return cls.postprocess(intermediate_text)
