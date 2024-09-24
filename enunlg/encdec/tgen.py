from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import logging
import tarfile

import omegaconf
import torch

from enunlg.encdec.seq2seq import BasicLSTMEncoder, LSTMDecWithAttention

if TYPE_CHECKING:
    import enunlg.vocabulary

logger = logging.getLogger(__name__)


class TGenEnc(BasicLSTMEncoder):
    def __init__(self, input_vocab_size, num_hidden_dims):
        """
        TGen uses an LSTM encoder and embeddings with the same dimensionality as the hidden layer.
        :param input_vocab_size:
        :param num_hidden_dims:
        """
        super(TGenEnc, self).__init__(input_vocab_size, num_hidden_dims, num_hidden_dims, init="zeros")

    @property
    def input_vocab_size(self):
        return self.num_unique_inputs

    @property
    def hidden_size(self):
        return self.num_hidden_dims


class TGenDec(LSTMDecWithAttention):
    pass


class TGenEncDec(torch.nn.Module):
    def __init__(self,
                 input_vocab_size: int,
                 output_vocab_size: int,
                 model_config=None):
        """
        TGenEncDec corresponds to TGen's seq2seq model with attention.

        TGen's model appears to be based on Tensorflow 0.6:
        https://github.com/tensorflow/tensorflow/blob/v0.6.0/tensorflow/python/ops/seq2seq.py
        """
        super().__init__()
        if model_config is None:
            # Set defaults
            model_config = omegaconf.DictConfig({'name': 'tgen',
                                                 'max_input_length': 30,
                                                 'encoder':
                                                     {'embeddings':
                                                         {'type': 'torch',
                                                          'num_embeddings': input_vocab_size,
                                                          'embedding_dim': 50,
                                                          'backprop': True
                                                          },
                                                      'cell': 'lstm',
                                                      'num_hidden_dims': 50},
                                                 'decoder':
                                                     {'embeddings':
                                                         {'type': 'torch',
                                                          'num_embeddings': output_vocab_size,
                                                          'dimensions': 50,
                                                          'backprop': True,
                                                          'padding_idx': 0,
                                                          'start_token_idx': 1,
                                                          'stop_token_idx': 2,
                                                          },
                                                      'cell': 'lstm',
                                                      'num_hidden_dims': 50
                                                      }
                                                 })
        self.config = model_config

        # Set basic properties
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        # Initialize encoder and decoder networks
        # TODO either tie enc-dec num hidden dims together or add code to handle config when they have diff dimensionality
        self.encoder = TGenEnc(self.input_vocab_size, self.config.encoder.num_hidden_dims)
        self.decoder = TGenDec(self.config.decoder.num_hidden_dims, self.output_vocab_size, self.config.max_input_length,
                               padding_idx=self.config.decoder.embeddings.get('padding_idx'),
                               start_token_idx=self.config.decoder.embeddings.get('start_idx'),
                               stop_token_idx=self.config.decoder.embeddings.get('stop_idx'))

        # Features copied from tgen.seq2seq.Seq2SeqBase
        # self.beam_size = 1
        # self.sample_top_k = 1
        # self.length_norm_weight = 0.0
        # self.context_bleu_weight = 0.0
        # self.context_bleu_metric = 'bleu'
        # self.slot_err_stats = None
        # self.classif_filter = None
        # self.lexicalizer = None
        # self.init_slot_err_stats()

        #
        # Attributes based on tgen.seq2seq.Seq2SeqGen
        # self.emb_size = 50
        # self.batch_size = 10
        # self.dropout_keep_prob = 1
        # self.optimizer_type = 'adam'
        #
        # self.passes = 5
        # self.min_passes = 1
        # self.improve_interval = 10
        # self.top_k = 5
        # self.use_dec_cost = False
        #
        # self.validation_size = 0
        # self.validation_freq = 10
        # self.validation_use_all_refs = False
        # self.validation_delex_slots = set()
        # self.validation_use_train_refs = False
        # self.multiple_refs = False
        # self.ref_selectors = None
        # self.max_cores = None
        # self.mode = 'tokens'
        # self.nn_type = 'emb_seq2seq'
        # self.randomize = True
        # self.cell_type = 'lstm'
        # self.bleu_validation_weight = 0.0
        # self.use_context = False
        #
        # self.train_summary_dir = None

    def init_slot_err_stats(self):
        raise NotImplementedError

    def _init_training(self):
        """
        In TGen's Seq2SeqGen class, performs further initialization for the class, including:
        * loading trees and dialogue acts from files for training & validation data
          * shrinking the training data if desired
        * extracting "embeddings" for DAs and texts (i.e. creating token-to-integer mappings for all of them, potentially including preprocessing like lowercasing)
        * calculating dimensions for tensors (i.e. vocab sizes and max DA/utterance lengths)
        * create batches
        * set up a lexicaliser
        * set up a classifier filter (?)
        * initialize costs and the NN structure
        """
        raise NotImplementedError()

    def encode(self, enc_emb: torch.Tensor):
        """
        Run encoding for a single set of integer inputs.

        :param enc_emb: tensor of integer inputs (shape?)
        :return: enc_outputs, enc_h_c_state (final state of the encoder)
        """
        enc_h_c_state = self.encoder.initial_h_c_state()
        # enc_outputs, enc_h_c_state
        return self.encoder(enc_emb, enc_h_c_state)

    def forward(self, enc_emb: torch.Tensor, max_output_length: int = 50):
        enc_outputs, enc_h_c_state = self.encode(enc_emb)

        dec_h_c_state = enc_h_c_state
        dec_input = torch.tensor([[self.decoder.start_idx]])
        dec_outputs = [self.decoder.start_idx]

        for dec_index in range(max_output_length):
            dec_output, dec_h_c_state = self.decoder(dec_input, dec_h_c_state, enc_outputs)
            topv, topi = dec_output.data.topk(1)
            dec_outputs.append(topi.item())
            if topi.item() == self.decoder.stop_idx:
                break
            dec_input = topi.squeeze().detach()
        return dec_outputs

    def forward_with_teacher_forcing(self, enc_emb: torch.Tensor, dec_emb: torch.Tensor) -> torch.Tensor:
        enc_outputs, enc_h_c_state = self.encode(enc_emb)

        dec_h_c_state = enc_h_c_state
        dec_outputs = torch.zeros((len(dec_emb), self.output_vocab_size))
        # First symbol is the start symbol
        dec_outputs[0] = dec_emb[0]

        for dec_input_index, dec_input in enumerate(dec_emb[:-1]):
            dec_output, dec_h_c_state = self.decoder(dec_input, dec_h_c_state, enc_outputs)
            dec_outputs[dec_input_index+1] = dec_output
        return dec_outputs

    def train_step(self, enc_emb: torch.Tensor, dec_emb: torch.Tensor, optimizer, criterion):
        optimizer.zero_grad()

        dec_outputs = self.forward_with_teacher_forcing(enc_emb, dec_emb)

        # We should be able to vectorise the following
        dec_targets = torch.tensor([x.unsqueeze(0) for x in dec_emb])
        loss = criterion(dec_outputs, dec_targets)

        loss.backward()
        optimizer.step()
        # mean loss per word returned in order for losses for sents of diff lengths to be comparable
        return loss.item() / dec_emb.size()[0]

    def generate(self, enc_emb, max_length=50):
        return self.generate_greedy(enc_emb, max_length)

    def generate_beam(self, enc_emb, max_length=50, beam_size=10, num_expansions: Optional[int] = None):
        if num_expansions is None:
            num_expansions = beam_size
        with torch.no_grad():
            enc_outputs, enc_h_c_state = self.encode(enc_emb)

            dec_h_c_state: torch.Tensor = enc_h_c_state
            prev_beam: List[Tuple[float, Tuple[int, ...], torch.Tensor]] = [(0.0, (1, ), dec_h_c_state)]

            for dec_index in range(max_length - 1):
                curr_beam = []
                for prev_beam_prob, prev_beam_item, prev_beam_hidden_state in prev_beam:
                    prev_item_index = prev_beam_item[-1]
                    if prev_item_index in (2, 0):
                        curr_beam.append((prev_beam_prob, prev_beam_item, prev_beam_hidden_state))
                    else:
                        dec_input = torch.tensor([[prev_item_index]])
                        dec_output, dec_h_c_state = self.decoder.forward(dec_input, prev_beam_hidden_state, enc_outputs)
                        top_values, top_indices = dec_output.data.topk(num_expansions)

                        for prob, candidate in zip(top_values, top_indices):
                            curr_beam.append((prev_beam_prob + float(prob), prev_beam_item + (int(candidate), ), dec_h_c_state))
                prev_beam = []
                prev_beam_set = set()
                # print(len(curr_beam))
                for prob, item, hidden in curr_beam:
                    if (prob, item) not in prev_beam_set:
                        prev_beam_set.add((prob, item))
                        prev_beam.append((prob, item, hidden))
                # print(len(prev_beam))
                prev_beam = sorted(prev_beam, key=lambda x: x[0] / len(x[1]), reverse=True)[:beam_size]
                # print(len(prev_beam))
            return [(prob/len(seq), seq) for prob, seq, _ in prev_beam]

    def generate_greedy(self, enc_emb: torch.Tensor, max_length=50):
        with torch.no_grad():
            return self.forward(enc_emb, max_length)

    def _save_classname_to_dir(self, directory_path):
        with (Path(directory_path) / "__class__.__name__").open('w') as class_file:
            class_file.write(self.__class__.__name__)

    def save(self, filepath, tgz=True):
        Path(filepath).mkdir()
        self._save_classname_to_dir(filepath)
        with (Path(filepath) / "_state_dict.pt").open('wb') as state_file:
            torch.save(self.state_dict(), state_file)
        with (Path(filepath) / "model_config.yaml").open('w') as config_file:
            omegaconf.OmegaConf.save(self.config, config_file)
        with (Path(filepath) / "_init_args.yaml").open('w') as init_args_file:
            omegaconf.OmegaConf.save({'input_vocab_size': self.input_vocab_size,
                                      'output_vocab_size': self.output_vocab_size},
                                     init_args_file)
        if tgz:
            with tarfile.open(f"{filepath}.tgz", mode="x:gz") as out_file:
                out_file.add(filepath, arcname=Path(filepath).parent)

    @classmethod
    def load_from_dir(cls, filepath):
        with (Path(filepath) / "__class__.__name__").open('r') as class_file:
            assert class_file.read().strip() == cls.__name__
        model_config = omegaconf.OmegaConf.load(Path(filepath) / 'model_config.yaml')
        init_args = omegaconf.OmegaConf.load(Path(filepath) / '_init_args.yaml')
        state_dict = torch.load(Path(filepath) / '_state_dict.pt')
        new_model = cls(input_vocab=enunlg.vocabulary.IntegralInformVocabulary([]),
                        output_vocab=enunlg.vocabulary.TokenVocabulary([]),
                        model_config=model_config)
        new_model.load_state_dict(state_dict)
        return new_model
