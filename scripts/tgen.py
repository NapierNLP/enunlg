"""Script for training, generating from, and evaluating TGen models."""

from typing import Tuple

import logging
import os
import random

from hydra.core.hydra_config import HydraConfig

import hydra
import matplotlib.pyplot as plt
import omegaconf
import seaborn as sns
import torch

from enunlg.normalisation.tokenisation import TGenTokeniser

import enunlg.data_management.e2e_challenge as e2e
import enunlg.embeddings.onehot as onehot
import enunlg.encdec.seq2seq as s2s
import enunlg.meaning_representation.dialogue_acts as das
import enunlg.trainer
import enunlg.util
import enunlg.vocabulary

MAX_INPUT_LENGTH_IN_KV_PAIRS = 10
MAX_INPUT_LENGTH_IN_INDICES = 3 * MAX_INPUT_LENGTH_IN_KV_PAIRS


def prep_tgen_integer_reps(input_corpus: e2e.E2ECorpus) -> Tuple[
    enunlg.vocabulary.IntegralInformVocabulary, enunlg.vocabulary.TokenVocabulary]:
    """
    Expects a corpus in E2E challenge format. Returns the "embeddings" for the input and output vocabularies
    (i.e. we create an integer-to-token reversible mapping for both separately)
    """
    mr_int_mapper = enunlg.vocabulary.IntegralInformVocabulary([mr for mr, _ in input_corpus])
    text_int_mapper = enunlg.vocabulary.TokenVocabulary([text.strip().split() for _, text in input_corpus])
    return mr_int_mapper, text_int_mapper


SUPPORTED_DATASETS = {"e2e", "e2e-cleaned"}


def load_data_from_config(data_config) -> e2e.E2ECorpus:
    if data_config.corpus.name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {data_config.corpus.name}")
    if data_config.corpus.name == 'e2e':
        logging.info("Loading E2E Challenge Data...")
        return e2e.load_e2e(data_config.corpus.splits)
    elif data_config.corpus.name == 'e2e-cleaned':
        logging.info("Loading the Cleaned E2E Data...")
        return e2e.load_e2e(data_config.corpus.splits, original=False)
    else:
        raise ValueError("We can only load the e2e dataset right now.")


def preprocess_corpus_from_config(preprocessing_config, corpus_to_process) -> e2e.E2ECorpus:
    if preprocessing_config.text.normalise == 'tgen':
        corpus_to_process = e2e.E2ECorpus([e2e.E2EPair(pair.mr, TGenTokeniser.tokenise(pair.text)) for pair in corpus_to_process])
    if preprocessing_config.text.delexicalise:
        logging.info('Applying delexicalisation...')
        if preprocessing_config.text.delexicalise.mode == 'split_on_caps':
            logging.info(f'...splitting on capitals in values')
            logging.info(f"...delexicalising: {preprocessing_config.text.delexicalise.slots}")
            corpus_to_process = e2e.E2ECorpus([e2e.delexicalise_exact_matches(pair,
                                                                              fields_to_delex=preprocessing_config.text.delexicalise.slots)
                                               for pair in corpus_to_process])
        else:
            raise ValueError("We can only handle the mode where we also check splitting on caps for values right now.")
    if preprocessing_config.mr.ignore_order:
        logging.info("Sorting slot-value pairs in the MR to ignore order...")
        corpus_to_process.sort_mr_elements()
    return corpus_to_process


def display_data_statistics(corpus, mr_int_mapper, token_int_mapper, onehot_encoder, train_enc_embs, train_dec_embs, train_enc_onehots, train_tokens) -> None:
    logging.info("Some basic corpus stats for this data prepared for TGen...")
    mr_lengths = [len(mr_emb) for mr_emb in train_enc_embs]
    logging.info(f"MR lengths:   {min(mr_lengths)} min, {max(mr_lengths)} max, {sum(mr_lengths)/len(mr_lengths)} avg")
    text_lengths = [len(text) for text in train_tokens]
    logging.info(f"Text lengths: {min(text_lengths)} min, {max(text_lengths)} max, {sum(text_lengths)/len(text_lengths)} avg")
    logging.info(f"Our input vocabulary has {mr_int_mapper.max_index + 1} unique tokens")
    logging.info("Basic MR representations:")
    enunlg.util.log_sequence([mr for mr, _ in corpus[:10]], indent="... ")
    logging.info("The same MRs as lists of vocab indices:")
    enunlg.util.log_sequence(train_enc_embs[:10], indent="... ")
    logging.info("The same MRs as one-hot vectors:")
    enunlg.util.log_sequence(train_enc_onehots[:10], indent="... ")
    logging.info("and converting back from one-hot vectors:")
    enunlg.util.log_sequence([onehot_encoder.embedding_to_string(onehot_vector) for onehot_vector in train_enc_onehots[:10]], indent="... ")
    logging.info(f"The lengths of those embeddings are:\n{[len(emb) for emb in train_enc_embs[:10]]}")
    logging.info(f"Our output vocabulary has {token_int_mapper.max_index + 1} unique tokens")
    logging.info("The reference texts for those MRs:")
    enunlg.util.log_sequence(train_tokens[:10], indent="... ")
    logging.info("The same texts as lists of vocab indices")
    enunlg.util.log_sequence(train_dec_embs[:10], indent="... ")


@hydra.main(version_base=None, config_path='../config', config_name='tgen')
def train_tgen(config: omegaconf.DictConfig):
    hydra_managed_output_dir = HydraConfig.get().runtime.output_dir
    logging.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    seed = config.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    corpus = load_data_from_config(config.data)

    corpus = preprocess_corpus_from_config(config.preprocessing, corpus)

    logging.info("Preparing training data for PyTorch...")

    # Prepare mr/input integer representation
    mr_int_mapper, token_int_mapper = prep_tgen_integer_reps(corpus)
    # Fixed input length is necessary for the TGen attention layer to work
    train_enc_indices = [mr_int_mapper.get_ints_with_padding(mr, MAX_INPUT_LENGTH_IN_KV_PAIRS) for mr, _ in corpus]

    # Prepare onehot encoding for semantic completeness scoring
    multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr, _ in corpus]
    onehot_encoder = onehot.DialogueActEmbeddings(multi_da_mrs, collapse_values=False)
    train_enc_onehots = [onehot_encoder.embed_da(mr) for mr in multi_da_mrs]

    # Prepare text/output integer representation
    train_tokens = [text.strip().split() for _, text in corpus]
    train_dec_indices = [token_int_mapper.get_ints_with_right_padding(text.split()) for _, text in corpus]

    # Summarise the data
    display_data_statistics(corpus, mr_int_mapper, token_int_mapper, onehot_encoder, train_enc_indices, train_dec_indices, train_enc_onehots, train_tokens)

    # Prepare validation data
    dev_config = omegaconf.DictConfig({'corpus': {'name': 'e2e', 'splits': ['devset']}})
    dev_corpus = load_data_from_config(dev_config)
    dev_corpus = preprocess_corpus_from_config(config.preprocessing, dev_corpus)
    dev_enc_indices = [mr_int_mapper.get_ints_with_padding(mr, MAX_INPUT_LENGTH_IN_KV_PAIRS) for mr, _ in corpus]
    dev_dec_indices = [token_int_mapper.get_ints_with_right_padding(text.split()) for _, text in dev_corpus]

    logging.info(f"Preparing neural network using {config.pytorch.device=}")
    DEVICE = config.pytorch.device
    tgen = s2s.TGenEncDec(mr_int_mapper, token_int_mapper, model_config=config.model).to(DEVICE)

    training_pairs = [(torch.tensor(enc_indices, dtype=torch.long, device=DEVICE),
                       torch.tensor(dec_indices, dtype=torch.long, device=DEVICE))
                      for enc_indices, dec_indices in zip(train_enc_indices, train_dec_indices)]
    validation_pairs = [(torch.tensor(enc_indices, dtype=torch.long, device=DEVICE),
                         torch.tensor(dec_indices, dtype=torch.long, device=DEVICE))
                        for enc_indices, dec_indices in zip(dev_enc_indices, dev_dec_indices)]

    logging.info(f"Running {config.mode.train.num_epochs} epochs of {len(training_pairs)} iterations (looking at each training pair once per epoch)")
    trainer = enunlg.trainer.TGenTrainer(tgen, training_config=config.mode.train)
    losses_for_plotting = trainer.train_iterations(training_pairs, validation_pairs)
    torch.save(tgen.state_dict(), os.path.join(hydra_managed_output_dir, "trained-tgen-model.pt"))

    sns.lineplot(data=losses_for_plotting)
    plt.savefig(os.path.join(hydra_managed_output_dir, 'training-loss.png'))


if __name__ == "__main__":
    corpus = train_tgen()
