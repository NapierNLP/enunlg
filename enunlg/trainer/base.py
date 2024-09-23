import logging

from torch.utils.tensorboard import SummaryWriter

import torch

logger = logging.getLogger('enunlg-scripts.multitask_seq2seq+attn')


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
        self.base_learning_rate = self.config.learning_rate
        if self.config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_learning_rate)
        elif self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.base_learning_rate)
        else:
            raise ValueError(f"Unsupported choice of optimizer. Expecting 'adam' or 'sgd' but got {self.config.optimizer}")

        # Initialize scheduler for optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=self.config.learning_rate_decay)
        self.tb_writer = SummaryWriter()

    def log_training_loss(self, loss, index):
        self.tb_writer.add_scalar(f'{self.__class__.__name__}-training_loss', loss, index)

    def log_parameter_gradients(self, index):
        for param, value in self.model.named_parameters():
            self.tb_writer.add_scalar(f"{self.__class__.__name__}-{param}-grad", torch.mean(value.grad), index)

    def train_iterations(self, *args, **kwargs):
        # TODO increase consistency between SCLSTM and TGen training so we can pull things up to this level
        raise NotImplementedError("Use one of the subclasses, don't try to use this one directly")

    def _log_epoch_begin_stats(self):
        logger.info(f"Learning rate is now {self.scheduler.get_last_lr()}")

    def _log_examples_this_interval(self, pairs):
        for i, o in pairs:
            logger.info("An example!")
            logger.info(f"Input:  {self.model.input_rep_to_string(i.tolist())}")
            logger.info(f"Ref:    {self.model.output_rep_to_string(o.tolist())}")
            logger.info(f"Output: {self.model.output_rep_to_string(self.model.generate(i))}")
