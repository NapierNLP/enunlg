import logging
import random
import time

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import enunlg.embeddings.onehot

import omegaconf
import torch
import torch.nn

import enunlg.encdec.seq2seq as s2s


class TGenSemClassifier(torch.nn.Module):
    def __init__(self, text_vocabulary: "enunlg.vocabulary.TokenVocabulary",
                 onehot_encoder: "enunlg.embeddings.onehot.DialogueActEmbeddings",
                 model_config=None) -> None:
        super().__init__()
        if model_config is None:
            # Set defaults
            model_config = omegaconf.DictConfig({'name': 'tgen_classifier',
                                    'max_mr_length': 30,
                                    'text_encoder':
                                        {'embeddings':
                                            {'mode': 'random',
                                             'dimensions': 50,
                                             'backprop': True
                                             },
                                         'cell': 'lstm',
                                         'num_hidden_dims': 128}
                                    })
        self.config = model_config

        self.text_vocabulary = text_vocabulary
        self.onehot_encoder = onehot_encoder

        self.text_encoder = s2s.TGenEnc(self.text_vocabulary.max_index + 1, self.num_hidden_dims, self.config.text_encoder.embeddings.dimensions)
        self.classif_linear = torch.nn.Linear(self.num_hidden_dims, self.onehot_encoder.dimensionality)
        self.classif_sigmoid = torch.nn.Sigmoid()

        # Initialise optimisers (same as in TGenEncDec model)
        self.learning_rate = 0.0005
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

    @property
    def num_hidden_dims(self):
        return self.config.text_encoder.num_hidden_dims

    def forward(self, input_text_ints):
        enc_h_c_state = self.text_encoder.initial_h_c_state()
        enc_outputs, _ = self.text_encoder(input_text_ints, enc_h_c_state)
        output = self.classif_linear(enc_outputs.squeeze(0)[-1])
        output = self.classif_sigmoid(output)
        return output

    def train_step(self, text_ints, mr_onehot):
        criterion = torch.nn.CrossEntropyLoss()
        self.optimizer.zero_grad()

        loss = 0.0
        output = self.forward(text_ints)
        loss += criterion(output.squeeze(0).squeeze(0), mr_onehot)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_iterations(self, pairs, epochs: int, record_interval: int = 1000) -> List[float]:
        """
        Run `epochs` training epochs over all training pairs, shuffling pairs in place each epoch.

        :param pairs: input and output indices for embeddings
        :param epochs: number of training epochs to run
        :param record_interval: how frequently to print and record training loss
        :return: list of average loss for each `record_interval` for each epoch
        """
        # In TGen this would begin with a call to self._init_training()

        start_time = time.time()
        prev_chunk_start_time = start_time
        loss_this_chunk = 0
        loss_to_plot = []

        for epoch in range(epochs):
            logging.info(f"Beginning epoch {epoch}...")
            logging.info(f"Learning rate is now {self.learning_rate}")
            random.shuffle(pairs)
            for index, (text_ints, mr_onehot) in enumerate(pairs, start=1):
                loss = self.train_step(text_ints, mr_onehot)
                loss_this_chunk += loss
                if index % record_interval == 0:
                    avg_loss = loss_this_chunk / record_interval
                    loss_this_chunk = 0
                    logging.info("------------------------------------")
                    logging.info(f"{index} iteration loss = {avg_loss}")
                    logging.info(f"Time this chunk: {time.time() - prev_chunk_start_time}")
                    prev_chunk_start_time = time.time()
                    loss_to_plot.append(avg_loss)
                    for i, o in pairs[:10]:
                        logging.info("An example!")
                        logging.info(f"Text:   {' '.join([x for x in self.text_vocabulary.get_tokens(i.tolist(), as_string=False) if x != '<VOID>'])}")
                        logging.info(f"MR:     {self.onehot_encoder.embedding_to_string(o.tolist())}")
                        prediction = self.predict(i).squeeze(0).squeeze(0).tolist()
                        output_list = [1.0 if x > 0.95 else 0.0 for x in prediction]
                        logging.info(f"Output: {self.onehot_encoder.embedding_to_string(output_list)}")
                        logging.info(f"One-hot target: {o.tolist()}")
                        logging.info(f"Current output: {prediction}")
            self.scheduler.step()
            logging.info("============================================")
        logging.info("----------")
        logging.info(f"Training took {(time.time() - start_time) / 60} minutes")
        return loss_to_plot

    def predict(self, text_ints):
        with torch.no_grad():
            return self.forward(text_ints)