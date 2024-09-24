from pathlib import Path

import logging

from sacrebleu import metrics as sm

try:
    import bert_score
except ModuleNotFoundError:
    bert_score = None
import omegaconf
import hydra
import torch

from enunlg.convenience.binary_mr_classifier import FullBinaryMRClassifier
from enunlg.data_management.loader import load_data_from_config
from enunlg.generators.multitask_seq2seq import MultitaskSeq2SeqGenerator, SingleVocabMultitaskSeq2SeqGenerator

import enunlg.data_management.enriched_e2e
import enunlg.data_management.enriched_webnlg
import enunlg.data_management.pipelinecorpus
import enunlg.encdec.multitask_seq2seq
import enunlg.trainer.multitask_seq2seq
import enunlg.util
import enunlg.vocabulary

logger = logging.getLogger('enunlg-scripts.analyse_corpus')

SUPPORTED_DATASETS = {"enriched-e2e", "enriched-webnlg"}


def evaluate(text_corpus, slot_value_corpus=None, ser_classifier=None) -> enunlg.data_management.pipelinecorpus.TextPipelineCorpus:
    # TODO rewrite this so we can have multiple refs
    relexed_best = [entry['best_output_relexed'] for entry in text_corpus]
    relexed_refs = [entry['ref_relexed'] for entry in text_corpus]

    # Calculate BLEU compared to targets
    bleu = sm.BLEU()
    # We only have one reference per output
    bleu_score = bleu.corpus_score(relexed_best, [relexed_refs])
    logger.info(f"Current score: {bleu_score}")
    text_corpus.metadata['BLEU'] = str(bleu_score)
    text_corpus.metadata['BLEU_settings'] = bleu.get_signature()

    if ser_classifier is None or slot_value_corpus is None:
        pass
    else:
        multi_da_mrs = ser_classifier.prepare_input(slot_value_corpus)
        # Estimate SER using classifier
        test_tokens = [text.strip().split() for text in relexed_best]
        test_text_ints = [ser_classifier.text_vocab.get_ints(text) for text in test_tokens]
        test_mr_bitvectors = [ser_classifier.binary_mr_vocab.embed_da(mr) for mr in multi_da_mrs]
        ser_pairs = [(torch.tensor(text_ints, dtype=torch.long),
                      torch.tensor(mr_bitvectors, dtype=torch.float))
                     for text_ints, mr_bitvectors in zip(test_text_ints, test_mr_bitvectors)]

        logger.info(f"Test error: {ser_classifier.evaluate(ser_pairs):0.2f}")
        text_corpus.metadata['SER'] = f"{ser_classifier.evaluate(ser_pairs):0.2f}"

    if bert_score is not None:
        (p, r, f1), bs_hash = bert_score.score(relexed_best, relexed_refs, return_hash=True, rescale_with_baseline=True,
                                               lang='en', verbose=True, device="cuda:0")
        logger.info(f"BERTScore: {p.mean()} / {r.mean()} / {f1.mean()}")
        text_corpus.metadata['BERTScore'] = f"BERTScore: {p.mean()} / {r.mean()} / {f1.mean()}"
        text_corpus.metadata['BERTScore_settings'] = bs_hash
    return text_corpus


@hydra.main(version_base=None, config_path='../config', config_name='analyse_corpus')
def analyse_corpus(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    enunlg.util.set_random_seeds(config.random_seed)

    text_corpus = enunlg.data_management.pipelinecorpus.TextPipelineCorpus.load(config.test.corpus_file)
    text_corpus.print_summary_stats()
    text_corpus.print_sample()
    # sv_corpus = enunlg.data_management.enriched_e2e.EnrichedE2ECorpus.from_text_corpus(text_corpus)
    # ser_classifier = FullBinaryMRClassifier.load(config.test.classifier_file)
    output_corpus = evaluate(text_corpus)  # , sv_corpus, ser_classifier)
    output_corpus.save(Path(config.output_dir) / "evaluation-with-scores.corpus")


if __name__ == "__main__":
    analyse_corpus()
