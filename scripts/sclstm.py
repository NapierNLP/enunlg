"""Script for training, generating from, and evaluating SC-LSTM models."""

import logging
import os


import hydra
import omegaconf
import torch

from enunlg.data_management.loader import load_data_from_config

import enunlg.data_management.cued as cued
import enunlg.embeddings.binary as onehot
import enunlg.encdec.sclstm as sclstm_models
import enunlg.meaning_representation.dialogue_acts as da_lib
import enunlg.normalisation.norms as norms
import enunlg.trainer
import enunlg.trainer.sclstm
import enunlg.util
import enunlg.vocabulary

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"sfx-restaurant", "e2e-cleaned"}


@hydra.main(version_base=None, config_path='../config', config_name='sclstm_as-released')
def sclstm_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir
        config.original_working_directory = hydra.utils.get_original_cwd()

    # Pass the config to the appropriate function depending on what mode we are using
    if config.mode == "train":
        train_sclstm(config)
    elif config.mode == "parameters":
        train_sclstm(config, shortcircuit="parameters")
    elif config.mode == "test":
        raise NotImplementedError("Testing mode for SCLSTM models not yet implemented")
        # test_sclstm(config)
    else:
        message = "Expected config.mode to specify `train` or `parameters` modes."
        raise ValueError(message)


def train_sclstm(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    corpus = load_data_from_config(config.data)
    corpus.print_summary_stats()
    print("____________")

    if config.data.corpus.name == "e2e-cleaned" and config.data.input_mode == "cued":
        logger.info("Converting E2E corpus to CUED corpus")
        corpus = [cued.CUEDPair(da_lib.MultivaluedDA.from_slot_value_list(act_type="inform", slot_values=pair.mr.items()), pair.text) for pair in corpus]
        corpus = cued.CUEDCorpus(corpus)

    if config.preprocessing.text.normalise == 'sclstm':
        # sclstm does normalisation before delexicalisation
        # SC-LSTM just uses existing whitespace, so the ExistingWhitespaceTokeniser is valid and doesn't need to do anything.
        for pair in corpus:
            for slot in pair.mr.slot_values:
                normed_vals = []
                values = pair.mr.slot_values[slot]
                if values == [None]:
                    continue
                else:
                    for value in values:
                        normed_vals.append(norms.SCLSTMNormaliser.normalise(value))
                    pair.mr.slot_values[slot] = normed_vals
            pair.text = norms.SCLSTMNormaliser.normalise(pair.text)
    if config.preprocessing.text.delexicalise:
        logger.info('Applying delexicalisation...')
        if config.preprocessing.text.delexicalise.mode == 'permutations':
            logger.info("...delexicalising all SFX-restaurant slots")
            # SC-LSTM delexicalises all slots it possibly can and tries different permutations of slots w/multiple values
            enunlg.util.log_sequence([mr for mr in corpus[:10]], indent="... ")
            corpus = cued.CUEDCorpus([cued.delexicalise_exact_matches(pair,
                                                                      cued.ALL_FIELDS,
                                                                      with_subscripts=False)
                                      for pair in corpus])
        else:
            raise ValueError("We can only handle the mode where we also check permutations of multiple values right now.")
    os.makedirs(os.path.join(config.original_working_directory, 'datasets', 'delexed-corpora'), exist_ok=True)
    with open(os.path.join(config.original_working_directory, 'datasets', 'delexed-corpora', f"{config.data.corpus.name}.{'_'.join(config.data.corpus.splits)}.sclstm-norm.sclstm-delex.txt"), 'w') as corpus_copy_file:
        for pair in corpus:
            corpus_copy_file.write(f"{pair.text}\n")

    da_embedder = onehot.DialogueActEmbeddings([mr for mr, _ in corpus])
    logger.info(f"Number of dimensions representing the dialogue act for an utterance and its slot-value pairs, respectively: {da_embedder.dialogue_act_size}, {da_embedder.slot_value_size}")
    train_tokens = [text.strip().split() for _, text in corpus]
    text_int_mapper = enunlg.vocabulary.TokenVocabulary(train_tokens)
    train_enc_embs = [da_embedder.embed_da(mr) for mr, _ in corpus]
    enunlg.util.log_sequence([mr for mr in corpus[:10]], indent="... ")
    enunlg.util.log_sequence([mr for mr in train_enc_embs[:10]], indent="... ")
    enunlg.util.log_sequence([da_embedder.embedding_to_string(mr) for mr in train_enc_embs[:10]])

    mr_size = len(train_enc_embs[0])
    text_lengths = [len(text) for text in train_tokens]
    logger.info(f"One-hot MR dimensions: {mr_size}")
    logger.info(f"Text lengths: {min(text_lengths)} min, {max(text_lengths)} max, {sum(text_lengths)/len(text_lengths)} avg")

    logger.info(f"Our output vocabulary has {text_int_mapper.max_index + 1} unique tokens")
    logger.info("The reference texts for those MRs:")
    enunlg.util.log_sequence(train_tokens[:10], indent="... ")

    logger.info(f"Preparing neural network using {config.pytorch.device=}")
    DEVICE = config.pytorch.device
    if config.model.embeddings.mode == 'glove':
        logger.info(f"Using GloVe embeddings from {config.model.embeddings.file}")
        sclstm = sclstm_models.SCLSTMModelAsReleasedWithGlove(da_embedder.size, config.model.embeddings.file, config.model).to(DEVICE)
        text_int_mapper = sclstm.output_vocab
    else:
        sclstm = sclstm_models.SCLSTMModelAsReleased(da_embedder.size, text_int_mapper.size, model_config=config.model)

    total_parameters = enunlg.util.count_parameters(sclstm)
    if shortcircuit == 'parameters':
        exit()

    train_dec_embs = []
    for _, text in corpus:
        train_dec_embs.append(text_int_mapper.get_ints(text.split()))
    logger.info("The same texts from above as lists of vocab indices")
    enunlg.util.log_sequence(train_dec_embs[:10], indent="... ")

    training_pairs = [(torch.tensor(enc_emb, dtype=torch.float, device=DEVICE),
                       torch.tensor(dec_emb, dtype=torch.long, device=DEVICE))
                      for enc_emb, dec_emb in zip(train_enc_embs, train_dec_embs)]

    logger.info(f"Running {config.train.num_epochs} epochs of {len(training_pairs)} iterations (looking at each training pair once per epoch)")
    # record_interval = 519 gives us 6 splits per epoch
    trainer = enunlg.trainer.sclstm.SCLSTMTrainer(sclstm, training_config=config.train)
    losses_for_plotting = trainer.train_iterations(training_pairs)
    torch.save(sclstm.state_dict(), os.path.join(config.output_dir, "trained-sclstm-model.pt"))


if __name__ == "__main__":
    corpus = sclstm_main()
