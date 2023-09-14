import logging
import random
import time

from typing import List, Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    pass

import box
import sacrebleu.metrics as sm
import torch


class BasicTrainer(object):
    def __init__(self,
                 model,
                 training_config):
        """
        A basic class to be implemented with particular details specified for different NN models

        :param model: a PyTorch NN model to be trained
        :param training_config: details for how the model should be trained and what we should track
        """
        self.model = model
        self.config = training_config
        self.epochs = self.config.num_epochs
        self.record_interval = self.config.record_interval
        self.shuffle_order_each_epoch = self.config.shuffle

        # Initialize loss
        # TODO add support for different loss functions
        self.loss = torch.nn.CrossEntropyLoss()

        # Initialize optimizers
        self.learning_rate = self.config.learning_rate
        if self.config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported choice of optimizer. Expecting 'adam' or 'sgd' but got {self.config.optimizer}")

        # Initialize scheduler for optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=self.config.learning_rate_decay)

    def train_iterations(self, **kwargs):
        # TODO increase consistency between SCLSTM and TGen training so we can pull things up to this level
        raise NotImplementedError("Use one of the subclasses, don't try to use this one directly")

    def _log_epoch_begin_stats(self):
        logging.info(f"Learning rate is now {self.learning_rate}")

    def _log_examples_this_interval(self, pairs):
        for i, o in pairs:
            logging.info("An example!")
            logging.info(f"Input:  {self.model.input_rep_to_string(i.tolist())}")
            logging.info(f"Ref:    {self.model.output_rep_to_string(o.tolist())}")
            logging.info(f"Output: {self.model.output_rep_to_string(self.model.generate(i))}")


class SCLSTMTrainer(BasicTrainer):
    def __init__(self,
                 model: "enunlg.encdec.sclstm.SCLSTMModel",
                 training_config=None):
        if training_config is None:
            # Set defaults
            training_config = box.Box({"num_epochs": 20,
                                    "record_interval": 519,
                                    "shuffle": True,
                                    "batch_size": 1,
                                    "optimizer": "sgd",
                                    "learning_rate": 0.1,
                                    "learning_rate_decay": 0.5
                                       })
        super().__init__(model, training_config)

        # Re-initialize loss using summation instead of mean
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def train_iterations(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                         teacher_forcing_rule=None) -> List[float]:
        """
        Run `epochs` training epochs over all training pairs, shuffling pairs in place each epoch.

        :param pairs: input and output indices for embeddings
        :param teacher_forcing_rule: what rule to use determining how much teacher forcing to use during training
        :return: list of average loss for each `record_interval` for each epoch
        """
        start_time = time.time()
        prev_chunk_start_time = start_time
        loss_this_interval = 0
        loss_to_plot = []

        prob_teacher_forcing = 1.0
        if teacher_forcing_rule is None:
            pass
        elif teacher_forcing_rule == 'reduce_over_scheduled_epochs':
            logging.info("Reducing teacher forcing linearly with number of (epochs / number of epochs scheduled)")
        else:
            logging.warning(f"Invalid value for teacher_forcing_rule: {teacher_forcing_rule}. Using default.")

        for epoch in range(self.epochs):
            if teacher_forcing_rule == 'reduce_over_scheduled_epochs':
                prob_teacher_forcing = 1.0 - epoch / self.epochs
            logging.info(f"Beginning epoch {epoch}...")
            self._log_epoch_begin_stats()
            random.shuffle(pairs)
            for index, (enc_emb, dec_emb) in enumerate(pairs, start=1):
                loss = self.model.train_step(enc_emb, dec_emb, self.optimizer, self.loss, prob_teacher_forcing)
                loss_this_interval += loss
                if index % self.record_interval == 0:
                    avg_loss = loss_this_interval / self.record_interval
                    loss_this_interval = 0
                    logging.info("------------------------------------")
                    logging.info(f"{index} iteration mean loss = {avg_loss}")
                    logging.info(f"Time this chunk: {time.time() - prev_chunk_start_time}")
                    prev_chunk_start_time = time.time()
                    loss_to_plot.append(avg_loss)
                    self._log_examples_this_interval(pairs[:10])
            self.scheduler.step()
            logging.info("============================================")
        logging.info("----------")
        logging.info(f"Training took {(time.time() - start_time) / 60} minutes")
        return loss_to_plot


class TGenTrainer(BasicTrainer):
    def __init__(self,
                 model: "enunlg.encdec.seq2seq.TGenEncDec",
                 training_config=None):
        if training_config is None:
            # Set defaults
            training_config = box.Box({"num_epochs": 20,
                                    "record_interval": 1000,
                                    "shuffle": True,
                                    "batch_size": 1,
                                    "optimizer": "adam",
                                    "learning_rate": 0.0005,
                                    "learning_rate_decay": 0.5 # TGen used 0.0
                                    })
        super().__init__(model, training_config)
        self._early_stopping_scores = [float('-inf')] * 5
        self._early_stopping_scores_changed = -1

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

        for epoch in range(self.epochs):
            logging.info(f"Beginning epoch {epoch}...")
            self._curr_epoch = epoch
            self._log_epoch_begin_stats()
            random.shuffle(pairs)
            for index, (enc_emb, dec_emb) in enumerate(pairs, start=1):
                loss = self.model.train_step(enc_emb, dec_emb, self.optimizer, self.loss)
                loss_this_interval += loss
                if index % self.record_interval == 0:
                    avg_loss = loss_this_interval / self.record_interval
                    loss_this_interval = 0
                    logging.info("------------------------------------")
                    logging.info(f"{index} iteration loss = {avg_loss}")
                    logging.info(f"Time this chunk: {time.time() - prev_chunk_start_time}")
                    prev_chunk_start_time = time.time()
                    loss_to_plot.append(avg_loss)
                    self._log_examples_this_interval(pairs[:10])
            if validation_pairs is not None:
                logging.info("Checking for early stopping!")
                # Add check for minimum number of passes
                if self.early_stopping_criterion_met(validation_pairs):
                    break
            self.scheduler.step()
            logging.info("============================================")
        logging.info("----------")
        logging.info(f"Training took {(time.time() - start_time) / 60} minutes")
        return loss_to_plot

    def early_stopping_criterion_met(self, validation_pairs):
        # TGen uses BLEU score for validation
        # Generate current realisations for MRs in validation pairs
        best_outputs = []
        ref_outputs = []
        for in_indices, out_indices in validation_pairs:
            # logging.info(f"Input:  {self.model.input_vocab.pretty_string(in_indices.tolist())}")
            # logging.info(f"Greedy: {self.model.output_vocab.pretty_string(self.model.generate_greedy(in_indices))}")
            # TGen does beam_size 10 and sets expansion size to be the same
            # (See TGen config.yaml line 35 and seq2seq.py line 219 `new_paths.extend(path.expand(self.beam_size, out_probs, st))`)
            cur_outputs = self.model.generate_beam(in_indices, beam_size=10, num_expansions=10)
            # The best output is the first one in the list, and the list contains pairs of length normalised logprobs along with the output indices
            best_outputs.append(self.model.output_vocab.pretty_string(cur_outputs[0][1]))
            ref_outputs.append(self.model.output_vocab.pretty_string(out_indices.tolist()))
        # Calculate BLEU compared to targets
        bleu = sm.BLEU()
        bleu_score = bleu.corpus_score(best_outputs, ref_outputs)
        logging.info(f"Current score: {bleu_score}")
        if bleu_score.score > self._early_stopping_scores[-1]:
            self._early_stopping_scores.append(bleu_score.score)
            self._early_stopping_scores = sorted(self._early_stopping_scores)[1:]
            self._early_stopping_scores_changed = self._curr_epoch
        # If BLEU score has changed recently, keep training
        # NOTE: right now we're using the length of _early_stopping_scores to effectively ensure a minimum of 5 epochs
        if self._curr_epoch - self._early_stopping_scores_changed < len(self._early_stopping_scores):
            return False
        # Otherwise, stop early
        else:
            logging.info("Scores have not improved recently on the validation set, so we are stopping training now.")
            return True
