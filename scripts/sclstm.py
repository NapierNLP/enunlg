"""Script for training, generating from, and evaluating SC-LSTM models."""

import logging
import os
import random

from hydra.core.hydra_config import HydraConfig

import hydra
import matplotlib.pyplot as plt
import omegaconf
import seaborn as sns
import torch

import enunlg.util

import enunlg.data_management.cued as cued
import enunlg.embeddings.onehot as onehot
import enunlg.encdec.sclstm as sclstm_models
import enunlg.meaning_representation.dialogue_acts as da_lib
import enunlg.normalisation.norms as norms
import enunlg.trainer
import enunlg.vocabulary

SUPPORTED_DATASETS = {"sfx-restaurant", "e2e-cleaned"}


@hydra.main(version_base=None, config_path='../config', config_name='sclstm_as-released')
def train_sclstm(config: omegaconf.DictConfig):
    original_working_dir = hydra.utils.get_original_cwd()
    hydra_managed_output_dir = HydraConfig.get().runtime.output_dir
    logging.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    seed = config.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if config.data.corpus.name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {config.data.corpus.name}")
    if config.data.corpus.name == 'sfx-restaurant':
        logging.info("Loading SFX Restaurant data...")
        corpus = cued.load_sfx_restaurant(config.data.corpus.splits)
    elif config.data.corpus.name == 'e2e-cleaned':
        logging.info("Loading E2E Challenge Corpus (cleaned)...")
        import enunlg.data_management.e2e_challenge as e2e
        logging.info("Converting E2E corpus to CUED corpus")
        corpus = cued.CUEDCorpus([cued.CUEDPair(da_lib.MultivaluedDA.from_slot_value_list(act_type="inform", slot_values=pair.mr.items()), pair.text) for pair in e2e.load_e2e(config.data.corpus.splits, original=False)])
    else:
        raise ValueError(f"We can only load the following datasets right now: {SUPPORTED_DATASETS}")
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
        logging.info('Applying delexicalisation...')
        if config.preprocessing.text.delexicalise.mode == 'permutations':
            logging.info(f"...delexicalising all SFX-restaurant slots")
            # SC-LSTM delexicalises all slots it possibly can and tries different permutations of slots w/multiple values
            enunlg.util.log_sequence([mr for mr in corpus[:10]], indent="... ")
            corpus = cued.CUEDCorpus([cued.delexicalise_exact_matches(pair,
                                                                      cued.ALL_FIELDS,
                                                                      with_subscripts=False)
                                      for pair in corpus])
        else:
            raise ValueError("We can only handle the mode where we also check permutations of multiple values right now.")
    os.makedirs(os.path.join(original_working_dir, 'datasets', 'delexed-corpora'), exist_ok=True)
    with open(os.path.join(original_working_dir, 'datasets', 'delexed-corpora', f"{config.data.corpus.name}.{'_'.join(config.data.corpus.splits)}.sclstm-norm.sclstm-delex.txt"), 'w') as corpus_copy_file:
        for pair in corpus:
            corpus_copy_file.write(f"{pair.text}\n")

    da_embedder = onehot.DialogueActEmbeddings([mr for mr, _ in corpus])
    logging.info(f"Number of dimensions representing the dialogue act for an utterance and its slot-value pairs, respectively: {da_embedder.dialogue_act_size}, {da_embedder.slot_value_size}")
    train_tokens = [text.strip().split() for _, text in corpus]
    text_int_mapper = enunlg.vocabulary.TokenVocabulary(train_tokens)
    train_enc_embs = [da_embedder.embed_da(mr) for mr, _ in corpus]
    enunlg.util.log_sequence([mr for mr in corpus[:10]], indent="... ")
    enunlg.util.log_sequence([mr for mr in train_enc_embs[:10]], indent="... ")
    enunlg.util.log_sequence([da_embedder.embedding_to_string(mr) for mr in train_enc_embs[:10]])

    mr_size = len(train_enc_embs[0])
    text_lengths = [len(text) for text in train_tokens]
    logging.info(f"One-hot MR dimensions: {mr_size}")
    logging.info(f"Text lengths: {min(text_lengths)} min, {max(text_lengths)} max, {sum(text_lengths)/len(text_lengths)} avg")

    logging.info(f"Our output vocabulary has {text_int_mapper.max_index + 1} unique tokens")
    logging.info("The reference texts for those MRs:")
    enunlg.util.log_sequence(train_tokens[:10], indent="... ")

    logging.info(f"Preparing neural network using {config.pytorch.device=}")
    DEVICE = config.pytorch.device
    if config.model.embeddings.mode == 'glove':
        logging.info(f"Using GloVe embeddings from {config.model.embeddings.file}")
        sclstm = sclstm_models.SCLSTMModelAsReleasedWithGlove(da_embedder, config.model.embeddings.file, config.model).to(DEVICE)
    else:
        sclstm = sclstm_models.SCLSTMModelAsReleased(da_embedder, text_int_mapper, model_config=config.model)
    train_dec_embs = []
    for _, text in corpus:
        train_dec_embs.append(sclstm.output_vocab.get_ints(text.split()))
    logging.info("The same texts from above as lists of vocab indices")
    enunlg.util.log_sequence(train_dec_embs[:10], indent="... ")

    training_pairs = [(torch.tensor(enc_emb, dtype=torch.float, device=DEVICE),
                       torch.tensor(dec_emb, dtype=torch.long, device=DEVICE))
                      for enc_emb, dec_emb in zip(train_enc_embs, train_dec_embs)]

    logging.info(f"Running {config.mode.train.num_epochs} epochs of {len(training_pairs)} iterations (looking at each training pair once per epoch)")
    # record_interval = 519 gives us 6 splits per epoch
    trainer = enunlg.trainer.SCLSTMTrainer(sclstm, training_config=config.mode.train)
    losses_for_plotting = trainer.train_iterations(training_pairs)
    torch.save(sclstm.state_dict(), os.path.join(hydra_managed_output_dir, "trained-sclstm-model.pt"))
    sns.lineplot(data=losses_for_plotting)
    plt.savefig(os.path.join(hydra_managed_output_dir, 'sclstm-training-loss.png'))


if __name__ == "__main__":
    corpus = train_sclstm()
