"""
Classes, functions, and constants for tokenisation.
"""

from typing import List

import abc
import logging

import regex

from enunlg.util import RegexRule

logger = logging.getLogger(__name__)


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
    """Class providing TGen's tokenisation rules for English texts."""
    # Three sets of rules. First set enforces spaces around punctuation.
    rules = (RegexRule(r'(([^\p{IsAlnum}\s\.\,−\-])\2*)', r' \1 '),
             RegexRule(r'([^\p{N}])([,.])([^\p{N}])', r'\1 \2 \3'),
             RegexRule(r'([^\p{N}])([,.])([\p{N}])', r'\1 \2 \3'),
             RegexRule(r'([\p{N}])([,.])([^\p{N}])', r'\1 \2 \3'),
             RegexRule(r'(–-)([^\p{N}])', r'\1 \2'),
             RegexRule(r'(\p{N} *|[^ ])(-)', r'\1\2 '),
             RegexRule(r' ([-−])', r'\1'),
             RegexRule(r' ([\'’´])', r'\1'),
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
    detok_rules = (RegexRule(r' (([^\p{IsAlnum}\s\.\,−\-])\2*) ', r'\1 '),
                   RegexRule(r'([^\p{N}]) ([,.]) ([^\p{N}])', r'\1\2 \3'),
                   RegexRule(r'([^\p{N}]) ([,.]) ([\p{N}])', r'\1\2 \3'),
                   RegexRule(r'([\p{N}]) ([,.]) ([^\p{N}])', r'\1\2 \3'),
                   RegexRule(r'(–-) ([^\p{N}])', r'\1\2'),
                   RegexRule(r'(\p{N} *|[^ ])(-) ', r'\1\2'),
                   RegexRule(r'([-−])', r' \1'),
                   # Second set keeps apostrophes together with words in most common contractions.
                   # RegexRule(r'([\'’´]) (s|m|d|ll|re|ve)\s', r' \1\2 '),
                   # RegexRule(r'(n [\'’´]) (t\s)', r' \1\2 '),
                   # Third set of contractions based on Treex.
                   RegexRule(r' ([Cc])an not ', r' \1annot '),
                   RegexRule(r' ([Gg])im me\s', r' \1imme '),
                   RegexRule(r' ([Gg])on na\s', r' \1onna '),
                   RegexRule(r' ([Gg])ot ta\s', r' \1otta '),
                   RegexRule(r' ([Ll])em me ', r' \1emme '),
                   RegexRule(r' ([Ww])an na ', r' \1anna '),
                   # Fourth set removes remaining spaces before punctuation
                   RegexRule(r' ([.,?!;:\'])', r'\1'),
                   RegexRule(r' (__[\p{IsAlnum}][\p{IsAlnum}]*) (-[\p{N}]__) ', r' \1\2 '),
             )

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

    @classmethod
    def detokenise(cls, text: str) -> str:
        intermediate_text = cls.preprocess(text)
        for rule in cls.detok_rules:
            intermediate_text = regex.sub(rule.match_expression, rule.replacement_expression, intermediate_text)
        return cls.postprocess(intermediate_text)


class INLG2024Tokenizer(AbstractTokeniser):
    """English tokenisation based mostly on TGen's tokenisation"""
    rules = (RegexRule(r'(([^\p{IsAlnum}\s\.\,−\-_])\2*)', r' \1 '),
             RegexRule(r'([^\p{N}])([,.])([^\p{N}])', r'\1 \2 \3'),
             RegexRule(r'([^\p{N}])([,.])([\p{N}])', r'\1 \2 \3'),
             RegexRule(r'([\p{N}])([,.])([^\p{N}])', r'\1 \2 \3'),
             RegexRule(r'(–-)([^\p{N}])', r'\1 \2'),
             RegexRule(r'(\p{N} *|[^ ])(-)', r'\1\2 '),
             RegexRule(r' ([-−])', r'\1'),
             RegexRule(r' ([\'’´])', r'\1'),
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
    detok_rules = (RegexRule(r' (([^\p{IsAlnum}\s\.\,−\-])\2*) ', r'\1 '),
                   RegexRule(r'([^\p{N}]) ([,.]) ([^\p{N}])', r'\1\2 \3'),
                   RegexRule(r'([^\p{N}]) ([,.]) ([\p{N}])', r'\1\2 \3'),
                   RegexRule(r'([\p{N}]) ([,.]) ([^\p{N}])', r'\1\2 \3'),
                   RegexRule(r'(–-) ([^\p{N}])', r'\1\2'),
                   RegexRule(r'(\p{N} *|[^ ])(-) ', r'\1\2'),
                   RegexRule(r'([-−])', r' \1'),
                   # Second set keeps apostrophes together with words in most common contractions.
                   # RegexRule(r'([\'’´]) (s|m|d|ll|re|ve)\s', r' \1\2 '),
                   # RegexRule(r'(n [\'’´]) (t\s)', r' \1\2 '),
                   # Third set of contractions based on Treex.
                   RegexRule(r' ([Cc])an not ', r' \1annot '),
                   RegexRule(r' ([Gg])im me\s', r' \1imme '),
                   RegexRule(r' ([Gg])on na\s', r' \1onna '),
                   RegexRule(r' ([Gg])ot ta\s', r' \1otta '),
                   RegexRule(r' ([Ll])em me ', r' \1emme '),
                   RegexRule(r' ([Ww])an na ', r' \1anna '),
                   # Fourth set removes remaining spaces before punctuation
                   RegexRule(r' ([.,?!;:\'])', r'\1'),
                   RegexRule(r' (__[\p{IsAlnum}][\p{IsAlnum}]*) (-[\p{N}]__) ', r' \1\2 '),
             )

    @classmethod
    def preprocess(cls, text: str):
        """TGen inserts spaces around text for easier regexes"""
        return f" {text} "

    @classmethod
    def postprocess(cls, toks: str):
        """TGen removes extra spaces from the text, so spaces can be used as token separators."""
        toks = regex.sub(r'\s+', ' ', toks).strip().split()
        retval = []
        curr_tok = ""
        for tok in toks:
            if "__" in tok:
                if tok.startswith("__") and not tok.endswith("__"):
                    curr_tok = tok
                elif tok.endswith("__") and not tok.startswith("__"):
                    if curr_tok:
                        retval.append(curr_tok+tok)
                        curr_tok = ""
            else:
                retval.append(tok)
        return " ".join(retval)

    @classmethod
    def tokenise(cls, text: str) -> str:
        intermediate_text = cls.preprocess(text)
        for rule in cls.rules:
            intermediate_text = regex.sub(rule.match_expression, rule.replacement_expression, intermediate_text)
        return cls.postprocess(intermediate_text)

    @classmethod
    def detokenise(cls, text: str) -> str:
        intermediate_text = cls.preprocess(text)
        for rule in cls.detok_rules:
            intermediate_text = regex.sub(rule.match_expression, rule.replacement_expression, intermediate_text)
        return cls.postprocess(intermediate_text)

