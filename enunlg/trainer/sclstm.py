import logging
import random
import time

from typing import List, Tuple, TYPE_CHECKING

import omegaconf
import torch

from enunlg.trainer.base import BasicTrainer

logger = logging.getLogger('enunlg-scripts.multitask_seq2seq+attn')

if TYPE_CHECKING:
    import enunlg.encdec.sclstm


class SCLSTMTrainer(BasicTrainer):
    def __init__(self,
                 model: "enunlg.encdec.sclstm.SCLSTMModel",
                 training_config=None):
        if training_config is None:
            # Set defaults
            training_config = omegaconf.DictConfig({"num_epochs": 20,
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
            logger.info("Reducing teacher forcing linearly with number of (epochs / number of epochs scheduled)")
        else:
            logging.warning(f"Invalid value for teacher_forcing_rule: {teacher_forcing_rule}. Using default.")

        for epoch in range(self.epochs):
            if teacher_forcing_rule == 'reduce_over_scheduled_epochs':
                prob_teacher_forcing = 1.0 - epoch / self.epochs
            logger.info(f"Beginning epoch {epoch}...")
            self._log_epoch_begin_stats()
            random.shuffle(pairs)
            for index, (enc_emb, dec_emb) in enumerate(pairs, start=1):
                loss = self.model.train_step(enc_emb, dec_emb, self.optimizer, self.loss)
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
            self.scheduler.step()
            logger.info("============================================")
        logger.info("----------")
        logger.info(f"Training took {(time.time() - start_time) / 60} minutes")
        self.tb_writer.close()
        return loss_to_plot
