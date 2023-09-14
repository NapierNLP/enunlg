"""Script for running the TGen NLU classifier."""

import logging

import hydra
import matplotlib.pyplot as plt
import omegaconf
import regex
import seaborn as sns
import torch

import enunlg.data_management.e2e_challenge as e2e
import enunlg.meaning_representation.dialogue_acts as das
import enunlg.embeddings.onehot as onehot
import enunlg
import enunlg as binary_mr_classifier
import enunlg.util

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


def tokenize(text):
    """
    Tokenize the given text (i.e., insert spaces around all tokens)

    For this function:
    Copyright © 2014-2017 Institute of Formal and Applied Linguistics,
                      Charles University, Prague.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    toks = ' ' + text + ' '  # for easier regexes

    # enforce space around all punct
    toks = regex.sub(r'(([^\p{IsAlnum}\s\.\,−\-])\2*)', r' \1 ', toks)  # all punct (except ,-.)
    toks = regex.sub(r'([^\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. & no numbers
    toks = regex.sub(r'([^\p{N}])([,.])([\p{N}])', r'\1 \2 \3', toks)  # ,. preceding numbers
    toks = regex.sub(r'([\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. following numbers
    toks = regex.sub(r'(–-)([^\p{N}])', r'\1 \2', toks)  # -/– & no number following
    toks = regex.sub(r'(\p{N} *|[^ ])(-)', r'\1\2 ', toks)  # -/– & preceding number/no-space
    toks = regex.sub(r'([-−])', r' \1', toks)  # -/– : always space before

    # keep apostrophes together with words in most common contractions
    toks = regex.sub(r'([\'’´]) (s|m|d|ll|re|ve)\s', r' \1\2 ', toks)  # I 'm, I 've etc.
    toks = regex.sub(r'(n [\'’´]) (t\s)', r' \1\2 ', toks)  # do n't

    # other contractions, as implemented in Treex
    toks = regex.sub(r' ([Cc])annot\s', r' \1an not ', toks)
    toks = regex.sub(r' ([Dd]) \' ye\s', r' \1\' ye ', toks)
    toks = regex.sub(r' ([Gg])imme\s', r' \1im me ', toks)
    toks = regex.sub(r' ([Gg])onna\s', r' \1on na ', toks)
    toks = regex.sub(r' ([Gg])otta\s', r' \1ot ta ', toks)
    toks = regex.sub(r' ([Ll])emme\s', r' \1em me ', toks)
    toks = regex.sub(r' ([Mm])ore\'n\s', r' \1ore \'n ', toks)
    toks = regex.sub(r' \' ([Tt])is\s', r' \'\1 is ', toks)
    toks = regex.sub(r' \' ([Tt])was\s', r' \'\1 was ', toks)
    toks = regex.sub(r' ([Ww])anna\s', r' \1an na ', toks)

    # clean extra space
    toks = regex.sub(r'\s+', ' ', toks)
    toks = toks.strip()
    return toks


SUPPORTED_DATASETS = {"e2e", "e2e-cleaned"}


@hydra.main(config_path='../config', config_name='tgen_classifier')
def run_tgen(config: omegaconf.DictConfig):
    if config.data.corpus.name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {config.data.corpus.name}")
    if config.data.corpus.name == 'e2e':
        logging.info("Loading E2E Challenge Data...")
        corpus = e2e.load_e2e(config.data.corpus.splits)
    elif config.data.corpus.name == 'e2e-cleaned':
        logging.info("Loading the Cleaned E2E Data...")
        corpus = e2e.load_e2e(config.data.corpus.splits, original=False)
    else:
        raise ValueError("We can only load the e2e dataset right now.")
    if config.preprocessing.text.normalise == 'tgen':
        corpus = e2e.E2ECorpus([e2e.E2EPair(pair.mr, tokenize(pair.text)) for pair in corpus])
    if config.preprocessing.text.delexicalise:
        logging.info('Applying delexicalisation...')
        if config.preprocessing.text.delexicalise.mode == 'split_on_caps':
            logging.info(f'...splitting on capitals in values')
            logging.info(f"...delexicalising: {config.preprocessing.text.delexicalise.slots}")
            corpus = e2e.E2ECorpus([e2e.delexicalise_exact_matches(pair,
                                                                   fields_to_delex=config.preprocessing.text.delexicalise.slots)
                                    for pair in corpus])
        else:
            raise ValueError("We can only handle the mode where we also check splitting on caps for values right now.")
    if config.preprocessing.mr.ignore_order:
        logging.info("Sorting slot-value pairs in the MR to ignore order...")
        corpus.sort_mr_elements()

    logging.info("Preparing training data for PyTorch...")
    # Prepare mr/input integer representation
    token_int_mapper = prep_tgen_text_integer_reps(corpus)
    # Prepare onehot encoding
    multi_da_mrs = [das.MultivaluedDA.from_slot_value_list('inform', mr.items()) for mr, _ in corpus]
    onehot_encoder = onehot.DialogueActEmbeddings(multi_da_mrs, collapse_values=False)
    train_mr_onehots = [onehot_encoder.embed_da(mr) for mr in multi_da_mrs]
    # Prepare text/output integer representation
    train_tokens = [text.strip().split() for _, text in corpus]
    text_lengths = [len(text) for text in train_tokens]
    train_text_ints = [token_int_mapper.get_ints_with_left_padding(text.split()) for _, text in corpus]
    logging.info(f"Text lengths: {min(text_lengths)} min, {max(text_lengths)} max, {sum(text_lengths)/len(text_lengths)} avg")
    logging.info("MRs as one-hot vectors:")
    enunlg.util.log_sequence(train_mr_onehots[:10], indent="... ")
    logging.info("and converting back from one-hot vectors:")
    enunlg.util.log_sequence([onehot_encoder.embedding_to_string(onehot_vector) for onehot_vector in train_mr_onehots[:10]], indent="... ")
    logging.info(f"Text vocabulary has {token_int_mapper.max_index + 1} unique tokens")
    logging.info("The reference texts for these MRs:")
    enunlg.util.log_sequence(train_tokens[:10], indent="... ")
    logging.info("The same texts as lists of vocab indices")
    enunlg.util.log_sequence(train_text_ints[:10], indent="... ")


    logging.info(f"Preparing neural network using {config.pytorch.device=}")
    DEVICE = config.pytorch.device
    tgen_classifier = binary_mr_classifier.TGenSemClassifier(token_int_mapper, onehot_encoder).to(DEVICE)

    training_pairs = [(torch.tensor(enc_emb, dtype=torch.long, device=DEVICE),
                       torch.tensor(dec_emb, dtype=torch.float, device=DEVICE))
                      for enc_emb, dec_emb in zip(train_text_ints, train_mr_onehots)]

    logging.info(f"Running {config.mode.train.num_epochs} epochs of {len(training_pairs)} iterations (looking at each training pair once per epoch)")
    losses_for_plotting = tgen_classifier.train_iterations(training_pairs, config.mode.train.num_epochs)

    torch.save(tgen_classifier.state_dict(), "trained-tgen_classifier-model.pt")

    sns.lineplot(data=losses_for_plotting)
    plt.savefig('training-loss.tgen_classifier.png')


if __name__ == "__main__":
    corpus = run_tgen()
