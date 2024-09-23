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

logger = logging.getLogger('enunlg-scripts.multitask_seq2seq+attn')

SUPPORTED_DATASETS = {"enriched-e2e", "enriched-webnlg"}


def train_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    torch.device("cpu")
    enunlg.util.set_random_seeds(config.random_seed)

    corpus, slot_value_corpus, text_corpus = enunlg.data_management.loader.prep_pipeline_corpus(config.data, config.train.train_splits)
    dev_corpus, dev_slot_value_corpus, dev_text_corpus = enunlg.data_management.loader.prep_pipeline_corpus(config.data, config.train.dev_splits)

    if config.data.drop_intermediate_layers:
        for tmp_corpus in (corpus, slot_value_corpus, text_corpus, dev_corpus, dev_slot_value_corpus, dev_text_corpus):
            tmp_corpus.drop_layers(keep=('raw_input', 'raw_output'))

    # drop entries that are too long
    indices_to_drop = []
    for idx, entry in enumerate(dev_text_corpus):
        if len(entry['raw_input']) > config.model.max_input_length - 2:
            indices_to_drop.append(idx)
            break
    logger.info(f"Dropping {len(indices_to_drop)} entries from the validation set for having too long an input rep.")
    for idx in reversed(indices_to_drop):
        dev_corpus.pop(idx)
        dev_slot_value_corpus.pop(idx)
        dev_text_corpus.pop(idx)

    text_corpus.write_to_iostream((Path(config.output_dir) / "text_corpus.nlg").open('w'))
    dev_text_corpus.write_to_iostream((Path(config.output_dir) / "dev_text_corpus.nlg").open('w'))

    # generator = SingleVocabMultitaskSeq2SeqGenerator(text_corpus, config.model)
    generator = MultitaskSeq2SeqGenerator(text_corpus, config.model)
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()

    trainer = enunlg.trainer.multitask_seq2seq.MultiDecoderSeq2SeqAttnTrainer(generator.model, config.train,
                                                                              input_vocab=generator.vocabularies["raw_input"],
                                                                              output_vocab=generator.vocabularies["raw_output"])

    # Section to be commented out normally, but useful for testing on small datasets
    # tmp_train_size = 50
    # tmp_dev_size = 10
    # slot_value_corpus = slot_value_corpus[:tmp_train_size]
    # text_corpus = text_corpus[:tmp_train_size]
    # dev_slot_value_corpus = dev_slot_value_corpus[:tmp_dev_size]
    # dev_text_corpus = dev_text_corpus[:tmp_dev_size]

    input_embeddings, output_embeddings = generator.prep_embeddings(text_corpus, config.model.max_input_length - 2)
    task_embeddings = [[output_embeddings[layer][idx]
                        for layer in generator.layers[1:]]
                       for idx in range(len(input_embeddings))]
    multitask_training_pairs = list(zip(input_embeddings, task_embeddings))

    dev_input_embeddings, dev_output_embeddings = generator.prep_embeddings(dev_text_corpus, config.model.max_input_length - 2)
    dev_task_embeddings = [[dev_output_embeddings[layer][idx]
                            for layer in generator.layers[1:]]
                           for idx in range(len(dev_input_embeddings))]
    multitask_validation_pairs = list(zip(dev_input_embeddings, dev_task_embeddings))

    trainer.train_iterations(multitask_training_pairs, multitask_validation_pairs)

    generator.save(Path(config.output_dir) / f"trained_{generator.__class__.__name__}.nlg")

    ser_classifier = FullBinaryMRClassifier.load(config.test.classifier_file)
    logger.info("===============================================")
    logger.info("Calculating performance on the training data...")
    train_corpus_eval, ser_corpus = generator.generate_output_corpus(slot_value_corpus, text_corpus, ser_classifier)
    train_corpus_eval = evaluate(train_corpus_eval, ser_corpus, ser_classifier)
    train_corpus_eval.save(Path(config.output_dir) / 'trainset-eval.corpus')
    
    logger.info("===============================================")
    logger.info("Calculating performance on the validation data...")
    dev_corpus_eval, dev_ser_corpus = generator.generate_output_corpus(dev_slot_value_corpus, dev_text_corpus, ser_classifier)
    dev_corpus_eval = evaluate(dev_corpus_eval, dev_ser_corpus, ser_classifier)
    dev_corpus_eval.save(Path(config.output_dir) / 'devset-eval.corpus')


def test_multitask_seq2seq_attn(config: omegaconf.DictConfig, shortcircuit=None) -> None:
    enunlg.util.set_random_seeds(config.random_seed)

    corpus, slot_value_corpus, text_corpus = enunlg.data_management.loader.prep_pipeline_corpus(config.data, config.test.test_splits)

    # drop entries that are too long
    indices_to_drop = []
    for idx, entry in enumerate(text_corpus):
        if len(entry['raw_input']) > config.model.max_input_length - 2:
            indices_to_drop.append(idx)
            break
    logger.info(f"Dropping {len(indices_to_drop)} entries from the test set for having too long an input rep.")
    for idx in reversed(indices_to_drop):
        corpus.pop(idx)
        text_corpus.pop(idx)
        slot_value_corpus.pop(idx)

    generator = MultitaskSeq2SeqGenerator.load(config.test.generator_file)
    if 'metadata' not in dir(generator):
        generator.metadata = {}
    if 'loaded_model' not in generator.metadata:
        generator.metadata['loaded_model'] = config.test.generator_file
    total_parameters = enunlg.util.count_parameters(generator.model)
    if shortcircuit == 'parameters':
        exit()

    ser_classifier = FullBinaryMRClassifier.load(config.test.classifier_file)
    output_corpus, ser_corpus = generator.generate_output_corpus(slot_value_corpus, text_corpus, include_slot_value_corpus=True)
    output_corpus = evaluate(output_corpus, ser_corpus, ser_classifier)
    output_corpus.save(Path(config.output_dir) / 'evaluation-output.corpus')


def evaluate(text_corpus, slot_value_corpus, ser_classifier=None) -> enunlg.data_management.pipelinecorpus.TextPipelineCorpus:
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

    if ser_classifier is not None:
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


@hydra.main(version_base=None, config_path='../config', config_name='multitask_seq2seq+attn')
def multitask_seq2seq_attn_main(config: omegaconf.DictConfig) -> None:
    # Add Hydra-managed output dir to the config dictionary
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    hydra_managed_output_dir = hydra_config.runtime.output_dir
    logger.info(f"Logs and output will be written to {hydra_managed_output_dir}")
    with omegaconf.open_dict(config):
        config.output_dir = hydra_managed_output_dir

    # Pass the config to the appropriate function depending on what mode we are using
    if config.mode == "train":
        train_multitask_seq2seq_attn(config)
    elif config.mode == "parameters":
        train_multitask_seq2seq_attn(config, shortcircuit="parameters")
    elif config.mode == "test":
        test_multitask_seq2seq_attn(config)
    else:
        message = "Expected config.mode to specify `train` or `parameters` modes."
        raise ValueError(message)


if __name__ == "__main__":
    multitask_seq2seq_attn_main()
