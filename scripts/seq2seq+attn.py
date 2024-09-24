import logging
import os
import random

import omegaconf
import hydra
import torch

from sacrebleu import metrics as sm

import enunlg.encdec.seq2seq as s2s
import enunlg.trainer
import enunlg.trainer.seq2seq
import enunlg.util
import enunlg.vocabulary

logger = logging.getLogger(__name__)

LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class NNGenerator(object):
    def __init__(self, input_vocab, output_vocab, model_class, model_config):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.model = model_class(model_config)

    def generate_from_input_seq(self, input_seq):
        return self._output_bridge(self.model.generate(self._input_bridge(input_seq)))
        pass

    def _input_bridge(self, input_seq):
        """convert input into input appropriate for self.model"""

    def _output_bridge(self, output_seq):
        """Convert raw output of self.model into output"""
        pass


def generate_uppercasing_data(num_entries):
    items = []
    for _ in range(10000):
        lb = random.choice(range(26))
        ub = random.choice(range(lb, 26))
        items.append((" ".join(LOWERCASE[lb:ub]).split(),
                      " ".join(UPPERCASE[lb:ub]).split()))
    return items


def prep_embeddings(vocab1, vocab2, tokens):
    return [(torch.tensor(vocab1.get_ints_with_left_padding(x[0], 26-2), dtype=torch.long),
             torch.tensor(vocab2.get_ints(x[1]), dtype=torch.long)) for x in tokens]


@hydra.main(version_base=None, config_path='../config', config_name='seq2seq+attn')
def seq2seq_attn_main(config: omegaconf.DictConfig):
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    if config.mode == "train":
        train_seq2seq_attn(config)
    elif config.mode == "test":
        test_seq2seq_attn(config)
    elif config.mode == "parameters":
        train_seq2seq_attn(config, shortcircuit="parameters")
    else:
        message = "Expected config.mode to specify `train` or `parameters` modes."
        raise ValueError(message)


def train_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None):
    enunlg.util.set_random_seeds(config.random_seed)

    items = generate_uppercasing_data(10000)
    train, dev, test = items[:8000], items[8000:9000], items[9000:]

    logger.info(len(train))

    input_vocab = enunlg.vocabulary.TokenVocabulary(LOWERCASE)
    output_vocab = enunlg.vocabulary.TokenVocabulary(UPPERCASE)

    train_embeddings = prep_embeddings(input_vocab, output_vocab, train)
    dev_embeddings = prep_embeddings(input_vocab, output_vocab, dev)

    model = s2s.Seq2SeqAttn(input_vocab.size, output_vocab.size, model_config=config.model)
    total_parameters = enunlg.util.count_parameters(model)
    if shortcircuit == 'parameters':
        exit()

    trainer = enunlg.trainer.seq2seq.Seq2SeqAttnTrainer(model, training_config=config.train, input_vocab=input_vocab, output_vocab=output_vocab)

    trainer.train_iterations(train_embeddings, validation_pairs=dev_embeddings)

    torch.save(model, os.path.join(config.output_dir, 'seq2seq+attn.pt'))


def test_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None):
    enunlg.util.set_random_seeds(config.random_seed)

    items = generate_uppercasing_data(10000)
    _, _, test = items[:8000], items[8000:9000], items[9000:]

    input_vocab = enunlg.vocabulary.TokenVocabulary(LOWERCASE)
    output_vocab = enunlg.vocabulary.TokenVocabulary(UPPERCASE)

    test_embeddings = prep_embeddings(input_vocab, output_vocab, test)

    model = torch.load(config.test.model_file)
    total_parameters = enunlg.util.count_parameters(model)
    if shortcircuit == 'parameters':
        exit()

    test_input = [x[0] for x in test_embeddings]
    test_ref = [x[1] for x in test_embeddings]
    outputs = [model.generate(embedding) for embedding in test_input]

    best_outputs = [" ".join(output_vocab.get_tokens([int(x) for x in output])) for output in outputs]
    ref_outputs = [" ".join(output_vocab.get_tokens([int(x) for x in output])) for output in test_ref]

    # Calculate BLEU compared to targets
    bleu = sm.BLEU()
    # We only have one reference per output
    bleu_score = bleu.corpus_score(best_outputs, [ref_outputs])
    logger.info(f"Current score: {bleu_score}")



if __name__ == "__main__":
    seq2seq_attn_main()
