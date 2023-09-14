from typing import Iterable, List


def tgen_lowercase(sentence: Iterable[str]) -> List[str]:
    """
    Lowercase a word token, keeping X-* placeholders + select all-caps words intact.

    copied from tgen.embeddings.TokenEmbeddingSeq2SeqExtract._lowercase()
    """
    return [token
            if token is None or token in ['I', 'OK'] or token.startswith('X-')
            else token.lower()
            for token in sentence]