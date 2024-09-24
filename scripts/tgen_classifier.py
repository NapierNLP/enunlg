"""Script for running the TGen NLU classifier."""

from pathlib import Path

import json
import logging

import hydra
import omegaconf
import torch

from enunlg.convenience.binary_mr_classifier import FullBinaryMRClassifier
from enunlg.data_management.loader import load_data_from_config
from enunlg.normalisation.tokenisation import TGenTokeniser

import enunlg
import enunlg.data_management.e2e_challenge as e2e
import enunlg.embeddings.binary
import enunlg.meaning_representation.dialogue_acts as das
import enunlg.trainer.binary_mr_classifier
import enunlg.util
import enunlg.vocabulary
from util import rdf_to_sv_set

logger = logging.getLogger("enunlg-scripts.tgen_classifier")

MAX_INPUT_LENGTH_IN_KV_PAIRS = 10
MAX_INPUT_LENGTH_IN_INDICES = 3 * MAX_INPUT_LENGTH_IN_KV_PAIRS
HIDDEN_LAYER_SIZE = 50

EPOCHS = 20


def prep_tgen_text_integer_reps(input_corpus):
    """
    Expects a corpus in E2E challenge format. Returns the vocabulary the texts.
    (i.e. we create an integer-to-token reversible mapping for both separately)
    """
    return enunlg.vocabulary.TokenVocabulary([text.strip().split() for _, text in input_corpus])

SUPPORTED_DATASETS = {"e2e", "e2e-cleaned", "enriched-webnlg"}


def preprocess(corpus, preprocessing_config):
    if preprocessing_config.text.normalise == 'tgen':
        corpus = e2e.E2ECorpus([e2e.E2EPair(pair.mr, TGenTokeniser.tokenize(pair.text)) for pair in corpus])
    if preprocessing_config.text.delexicalise:
        logger.info('Applying delexicalisation...')
        if preprocessing_config.text.delexicalise.mode == 'split_on_caps':
            logger.info('...splitting on capitals in values')
            logger.info(f"...delexicalising: {preprocessing_config.text.delexicalise.slots}")
            corpus = e2e.E2ECorpus([e2e.delexicalise_exact_matches(pair,
                                                                   fields_to_delex=preprocessing_config.text.delexicalise.slots)
                                    for pair in corpus])
        else:
            message = "We can only handle the mode where we also check splitting on caps for values right now."
            raise ValueError(message)
    if preprocessing_config.mr.ignore_order:
        logger.info("Sorting slot-value pairs in the MR to ignore order...")
        corpus.sort_mr_elements()
    return corpus


def rejoin_sem_classes(text):
    out_list = []
    curr_token = ""
    for token in text.strip().split():
        if token == "__":
            if curr_token.startswith("__"):
                out_list.append(f"{curr_token}{token}")
                curr_token = ""
            elif curr_token == "":
                curr_token = token
        else:
            if curr_token == "":
                out_list.append(token)
            else:
                curr_token = f"{curr_token}{token}"
    return " ".join(out_list)


def webnlg_to_e2e(corpus):
    return e2e.E2ECorpus([e2e.E2EPair(rdf_to_sv_set(entry.raw_input),
                                      rejoin_sem_classes(TGenTokeniser.tokenize(entry.raw_output)))
                          for entry in corpus])
    

@hydra.main(version_base=None, config_path='../config', config_name='tgen_classifier')
def tgen_classifier_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    # Pass the config to the appropriate function depending on what mode we are using
    if config.mode == "train":
        train_tgen_classifier(config)
    elif config.mode == "parameters":
        train_tgen_classifier(config, shortcircuit="parameters")
    elif config.mode == "test":
        test_tgen_classifier(config)
    else:
        message = "Expected config.mode to specify `train` or `parameters` modes."
        raise ValueError(message)


def train_tgen_classifier(config: omegaconf.DictConfig, shortcircuit=None):
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data, config.train.train_splits)
    corpus.print_summary_stats()
    print("____________")
    dev_corpus = load_data_from_config(config.data, config.train.dev_splits)
    dev_corpus.print_summary_stats()
    print("____________")

    if config.data.corpus.name == "webnlg-enriched":
        sem_class_dict = json.load(Path("datasets/processed/enriched-webnlg.dbo-delex.70-percent-coverage.json").open('r'))
        sem_class_lower = {key.lower(): sem_class_dict[key] for key in sem_class_dict}
        corpus.delexicalise_with_sem_classes(sem_class_lower)
        dev_corpus.delexicalise_with_sem_classes(sem_class_lower)
        corpus = webnlg_to_e2e(corpus)
        dev_corpus = webnlg_to_e2e(dev_corpus)

    corpus = preprocess(corpus, config.preprocessing)
    dev_corpus = preprocess(dev_corpus, config.preprocessing)
    if config.data.corpus.name == "webnlg-enriched":
        for entry in corpus:
            entry.text = rejoin_sem_classes(entry.text)
        for entry in dev_corpus:
            entry.text = rejoin_sem_classes(entry.text)

    logger.info("Preparing training data for PyTorch...")
    # Prepare input integer representation
    token_int_mapper = prep_tgen_text_integer_reps(corpus)
    # Prepare bitvector encoding
    if config.data.corpus.name == "webnlg-enriched":
        multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', list(mr)) for mr, _ in corpus]
        dev_multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', list(mr)) for mr, _ in dev_corpus]
    else:
        multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr, _ in corpus]
        dev_multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr, _ in dev_corpus]
    bitvector_encoder = enunlg.embeddings.binary.DialogueActEmbeddings(multi_da_mrs, collapse_values=False)

    # Prepare text/output integer representation
    train_mr_bitvectors = [bitvector_encoder.embed_da(mr) for mr in multi_da_mrs]
    train_tokens = [text.strip().split() for _, text in corpus]
    text_lengths = [len(text) for text in train_tokens]
    train_text_ints = [token_int_mapper.get_ints_with_left_padding(text.split()) for _, text in corpus]
    logger.info(f"Text lengths: {min(text_lengths)} min, {max(text_lengths)} max, {sum(text_lengths)/len(text_lengths)} avg")
    logger.info("MRs as bitvectors:")
    enunlg.util.log_sequence(train_mr_bitvectors[:10], indent="... ")
    logger.info("and converting back from bitvectors:")
    enunlg.util.log_sequence([bitvector_encoder.embedding_to_string(bitvector) for bitvector in train_mr_bitvectors[:10]], indent="... ")
    logger.info(f"Text vocabulary has {token_int_mapper.max_index + 1} unique tokens")
    logger.info("The reference texts for these MRs:")
    enunlg.util.log_sequence(train_tokens[:10], indent="... ")
    logger.info("The same texts as lists of vocab indices")
    enunlg.util.log_sequence(train_text_ints[:10], indent="... ")

    classifier = FullBinaryMRClassifier(token_int_mapper, bitvector_encoder, config.model)
    total_parameters = enunlg.util.count_parameters(classifier.model)
    if shortcircuit == 'parameters':
        exit()

    trainer = enunlg.trainer.binary_mr_classifier.BinaryMRClassifierTrainer(classifier.model, config.train, token_int_mapper, bitvector_encoder)

    training_pairs = [(torch.tensor(enc_emb, dtype=torch.long),
                       torch.tensor(dec_emb, dtype=torch.float))
                      for enc_emb, dec_emb in zip(train_text_ints, train_mr_bitvectors)]
    dev_text_ints = [token_int_mapper.get_ints_with_left_padding(text.split()) for _, text in dev_corpus]
    dev_mr_bitvectors = [bitvector_encoder.embed_da(mr) for mr in dev_multi_da_mrs]
    validation_pairs = [(torch.tensor(enc_emb, dtype=torch.long),
                        torch.tensor(dec_emb, dtype=torch.float))
                        for enc_emb, dec_emb in zip(dev_text_ints, dev_mr_bitvectors)]

    logger.info(f"Running {config.train.num_epochs} epochs of {len(training_pairs)} iterations (with {len(validation_pairs)} validation pairs")
    losses_for_plotting = trainer.train_iterations(training_pairs, validation_pairs)

    classifier.save(Path(config.output_dir) / f'trained_{classifier.__class__.__name__}.nlg')


def test_tgen_classifier(config: omegaconf.DictConfig, shortcircuit=None):
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data, config.test.test_splits)
    corpus.print_summary_stats()
    print("____________")

    corpus = preprocess(corpus, config.preprocessing)


    classifier = FullBinaryMRClassifier.load(config.test.classifier_file)

    # Prepare text/output integer representation
    test_tokens = [text.strip().split() for _, text in corpus]
    test_text_ints = [classifier.text_vocab.get_ints_with_left_padding(text) for text in test_tokens]
    multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr, _ in corpus]
    test_mr_bitvectors = [classifier.binary_mr_vocab.embed_da(mr) for mr in multi_da_mrs]

    total_parameters = enunlg.util.count_parameters(classifier.model)
    if shortcircuit == 'parameters':
        exit()

    test_pairs = [(torch.tensor(text_ints, dtype=torch.long),
                   torch.tensor(mr_bitvectors, dtype=torch.float))
                  for text_ints, mr_bitvectors in zip(test_text_ints, test_mr_bitvectors)]
    error = classifier.evaluate(test_pairs)
    logger.info(f"Test error: {error:0.2f}")


if __name__ == "__main__":
    tgen_classifier_main()
