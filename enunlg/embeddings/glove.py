import torch
import torch.nn

import enunlg.vocabulary


class GloVeEmbeddings(torch.nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    @staticmethod
    def from_word_embedding_txt(filepath, with_vocab=False):
        tokens = []
        embeddings = []
        with open(filepath, 'r') as embedding_file:
            for line in embedding_file:
                word_embedding = line.strip().split()
                tokens.append(word_embedding[0])
                embeddings.append(word_embedding[1:])
        vocab = enunlg.vocabulary.TokenVocabulary([tokens])
        matrix_size = (len(vocab.tokens), len(embeddings[0]))
        matrix = torch.rand(matrix_size)
        for token, embedding in zip(tokens, embeddings):
            matrix[vocab.get_int(token)] = torch.tensor([float(x) for x in embedding])
        glove_embeddings = GloVeEmbeddings.from_pretrained(matrix)
        if with_vocab:
            return vocab, glove_embeddings
        else:
            return glove_embeddings

    @staticmethod
    def from_word_embedding_txt_with_wen_etal_header(filepath, with_vocab=False):
        tokens = []
        embeddings = []
        with open(filepath, 'r') as embedding_file:
            # Skip header
            for _ in range(5):
                next(embedding_file)
            for line in embedding_file:
                word_embedding = line.strip().split()
                tokens.append(word_embedding[0])
                embeddings.append(word_embedding[1:])
        vocab = enunlg.vocabulary.TokenVocabulary([tokens])
        matrix_size = (len(vocab.tokens), len(embeddings[0]))
        matrix = torch.rand(matrix_size)
        for token, embedding in zip(tokens, embeddings):
            matrix[vocab.get_int(token)] = torch.tensor([float(x) for x in embedding])
        glove_embeddings = GloVeEmbeddings.from_pretrained(matrix)
        if with_vocab:
            return vocab, glove_embeddings
        else:
            return glove_embeddings


if __name__ == "__main__":
    try:
        vocab, embeddings = GloVeEmbeddings.from_word_embedding_txt('../../datasets/RNNLG/vec/vectors-80.txt', with_vocab=True)
        print(vocab)
        print(embeddings)
    except FileNotFoundError:
        raise FileNotFoundError("\nUnable to find the GloVe vectors from Wen et al."
                                "\nDid you already run scripts/fetch_cued.bash ?")
