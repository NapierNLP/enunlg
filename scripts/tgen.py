"""Script for training, generating from, and evaluating TGen models."""

from pathlib import Path

import logging

from hydra.core.hydra_config import HydraConfig

import hydra
import omegaconf
import torch

import enunlg.data_management.loader
import enunlg.embeddings.binary
import enunlg.encdec.tgen
import enunlg.generators.tgen
import enunlg.trainer
import enunlg.trainer.tgen
import enunlg.util
import enunlg.vocabulary

MAX_INPUT_LENGTH_IN_KV_PAIRS = 10
MAX_INPUT_LENGTH_IN_INDICES = 3 * MAX_INPUT_LENGTH_IN_KV_PAIRS

logger = logging.getLogger(__name__)


def train_tgen(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = enunlg.data_management.loader.prep_corpus(config.data, config.train.train_splits)
    dev_corpus = enunlg.data_management.loader.prep_corpus(config.data, config.train.dev_splits)

    generator = enunlg.generators.tgen.TGenGenerator(corpus, config.model)
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()

    trainer = enunlg.trainer.tgen.TGenTrainer(generator.model,
                                              training_config=config.train,
                                              input_vocab=generator.input_vocab,
                                              output_vocab=generator.output_vocab)

    # Fixed input length is necessary for the TGen attention layer to work
    train_enc_indices, train_dec_indices = generator.prep_embeddings(corpus, MAX_INPUT_LENGTH_IN_KV_PAIRS)
    dev_enc_indices, dev_dec_indices = generator.prep_embeddings(dev_corpus, MAX_INPUT_LENGTH_IN_KV_PAIRS)

    training_pairs = [(torch.tensor(enc_indices, dtype=torch.long),
                       torch.tensor(dec_indices, dtype=torch.long))
                      for enc_indices, dec_indices in zip(train_enc_indices, train_dec_indices)]
    validation_pairs = [(torch.tensor(enc_indices, dtype=torch.long),
                         torch.tensor(dec_indices, dtype=torch.long))
                        for enc_indices, dec_indices in zip(dev_enc_indices, dev_dec_indices)]

    losses_for_plotting = trainer.train_iterations(training_pairs, validation_pairs)
    generator.save(Path(config.output_dir) / f"trained_{generator.__class__.__name__}.nlg")


@hydra.main(version_base=None, config_path='../config', config_name='tgen')
def tgen_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir
    if config.mode == "train":
        train_tgen(config)
    elif config.mode == "parameters":
        train_tgen(config, shortcircuit="parameters")
    else:
        message = "Expected config.mode to specify `train` or `parameters` modes."
        raise ValueError(message)


if __name__ == "__main__":
    tgen_main()
