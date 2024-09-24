from typing import List, Tuple, Optional, TYPE_CHECKING

import logging
import random
import time

from tqdm import tqdm

import numpy as np
import omegaconf
import torch

from enunlg.trainer.base import BasicTrainer
from enunlg.util import hamming_error

logger = logging.getLogger('enunlg-scripts.multitask_seq2seq+attn')

if TYPE_CHECKING:
    import enunlg.nlu.binary_mr_classifier


class BinaryMRClassifierTrainer(BasicTrainer):
    def __init__(self,
                 model: "enunlg.nlu.binary_mr_classifier.TGenSemClassifier",
                 training_config=None,
                 text_vocab=None,
                 bitvector_vocab=None):
        if training_config is None:
            # Set defaults
            training_config = omegaconf.DictConfig({"num_epochs": 20,
                                                    "record_interval": 1000,
                                                    "shuffle": True,
                                                    "batch_size": 1,
                                                    "optimizer": "adam",
                                                    "learning_rate": 0.0005,
                                                    "learning_rate_decay": 0.5  # TGen used 0.0
                                                   })
        super().__init__(model, training_config)
        self.loss = torch.nn.BCELoss()
        self.text_vocab = text_vocab
        self.bitvector_vocab = bitvector_vocab
        self._curr_epoch = -1
        self._early_stopping_scores = [float('inf')] * 5
        self._early_stopping_scores_changed = 0

    def _log_examples_this_interval(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        for i, o in pairs[:10]:
            logger.info("An example!")
            logger.info(f"Text:   {' '.join([x for x in self.text_vocab.get_tokens(i.tolist()) if x != '<VOID>'])}")
            logger.info(f"MR:     {self.bitvector_vocab.embedding_to_string(o.tolist())}")
            prediction = self.model.predict(i).squeeze(0).squeeze(0).tolist()
            # output_list = [1.0 if x > 0.95 else 0.0 for x in prediction]
            # logger.info(f"Output: {self.onehot_encoder.embedding_to_string(output_list)}")
            logger.info(f"Output: {self.bitvector_vocab.embedding_to_string(list(np.round(prediction)))}")
            target_bitvector = np.round(o.tolist())
            output_bitvector = np.round(prediction)
            logger.info(f"Target bitvector: {target_bitvector}")
            logger.info(f"Output bitvector: {output_bitvector}")
            logger.info(f"Error: {hamming_error(target_bitvector, output_bitvector)}")

    def train_iterations(self,
                         pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                         validation_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> List[float]:
        """
        Run `epochs` training epochs over all training pairs, shuffling pairs in place each epoch.

        :param pairs: input and output indices for embeddings
        :param validation_pairs: input and output indices for embeddings to be used in the validation step
        :return: list of average loss for each `record_interval` for each epoch
        """
        start_time = time.time()
        prev_chunk_start_time = start_time
        loss_this_interval = 0
        loss_to_plot = []

        for epoch in range(1, self.epochs + 1):
            logger.info(f"Beginning epoch {epoch}...")
            self._curr_epoch = epoch
            self._log_epoch_begin_stats()
            random.shuffle(pairs)
            for index, (text_ints, mr_onehot) in tqdm(enumerate(pairs, start=1)):
                loss = self.model.train_step(text_ints, mr_onehot, self.optimizer, self.loss)
                self.log_training_loss(float(loss), epoch * len(pairs) + index)
                self.log_parameter_gradients(epoch * len(pairs) + index)
                loss_this_interval += loss
                if index % self.record_interval == 0:
                    avg_loss = loss_this_interval / self.record_interval
                    loss_this_interval = 0
                    logger.info("------------------------------------")
                    logger.info(f"{index} iteration mean loss = {avg_loss}")
                    logger.info(f"Time this chunk: {time.time() - prev_chunk_start_time}")
                    prev_chunk_start_time = time.time()
                    loss_to_plot.append(avg_loss)
                    self._log_examples_this_interval(pairs[:10])
                    self.tb_writer.flush()
            if validation_pairs is not None:
                logger.info("Checking for early stopping!")
                # Add check for minimum number of passes
                if self.early_stopping_criterion_met(validation_pairs):
                    break
            self.scheduler.step()
            logger.info("============================================")
        logger.info("----------")
        logger.info(f"Training took {(time.time() - start_time) / 60} minutes")
        self.tb_writer.close()
        return loss_to_plot

    def sample_generations_and_references(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[str], List[str]]:
        best_outputs = []
        ref_outputs = []
        for in_indices, out_indices in pairs:
            # TGen does beam_size 10 and sets expansion size to be the same
            # (See TGen config.yaml line 35 and seq2seq.py line 219 `new_paths.extend(path.expand(self.beam_size, out_probs, st))`)
            cur_outputs = self.model.generate_beam(in_indices, beam_size=10, num_expansions=10)
            # The best output is the first one in the list, and the list contains pairs of length normalised logprobs along with the output indices
            best_outputs.append(self.bitvector_vocab.pretty_string(cur_outputs[0][1]))
            ref_outputs.append(self.bitvector_vocab.pretty_string(out_indices.tolist()))
        return best_outputs, ref_outputs

    def early_stopping_criterion_met(self, validation_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        error = 0
        for i, o in validation_pairs:
            prediction = self.model.predict(i).squeeze(0).squeeze(0).tolist()
            target_bitvector = np.round(o.tolist())
            output_bitvector = np.round(prediction)
            error += hamming_error(target_bitvector, output_bitvector)
        error = error / len(validation_pairs)
        logger.info(f"Current error score: {error}")
        if error < self._early_stopping_scores[-1]:
            self._early_stopping_scores.append(error)
            self._early_stopping_scores = sorted(self._early_stopping_scores, reverse=True)[1:]
            self._early_stopping_scores_changed = self._curr_epoch
        # If error has changed recently, keep training
        # NOTE: right now we're using the length of _early_stopping_scores to effectively ensure a minimum of 5 epochs
        if self._curr_epoch - self._early_stopping_scores_changed < len(self._early_stopping_scores):
            return False
        # Otherwise, stop early
        else:
            logger.info("Scores have not improved recently on the validation set, so we are stopping training now.")
            return True
