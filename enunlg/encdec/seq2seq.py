from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass

import omegaconf
import torch
import torch.nn
import torch.nn.functional

DEVICE = torch.device("cpu")


class TGenEnc(torch.nn.Module):
    def __init__(self, input_vocab_size, num_hidden_dims, num_embedding_dims=None):
        """
        TGen uses an LSTM encoder and embeddings with the same dimensionality as the hidden layer.
        :param input_vocab_size:
        :param num_hidden_dims:
        """
        super(TGenEnc, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.hidden_size = num_hidden_dims
        if num_embedding_dims is None:
            self.embedding_size = self.hidden_size
        else:
            self.embedding_size = num_embedding_dims

        self.embedding = torch.nn.Embedding(self.input_vocab_size, self.embedding_size)
        self.lstm = torch.nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)

    def forward(self, input_indices, h_c_state):
        embedded = self.embedding(input_indices).view(1, len(input_indices), -1)
        return self.lstm(embedded, h_c_state)

    def forward_one_step(self, input_index, h_c_state):
        embedded = self.embedding(input_index).view(1, 1, -1)
        output, h_c_state = self.lstm(embedded, h_c_state)
        return output, h_c_state

    def initial_h_c_state(self):
        return (torch.zeros(1, 1, self.hidden_size, device=DEVICE),
                torch.zeros(1, 1, self.hidden_size, device=DEVICE))


class BasicDecoder(torch.nn.Module):
    def __init__(self, hidden_layer_size, output_vocab_size, max_input_length, num_embedding_dims=None):
        super(BasicDecoder, self).__init__()
        self.hidden_size = hidden_layer_size
        if num_embedding_dims == None:
            self.embedding_size = self.hidden_size
        else:
            self.embedding_size = num_embedding_dims
        self.output_vocab_size = output_vocab_size
        self.max_input_length = max_input_length

        self.output_embeddings = torch.nn.Embedding(self.output_vocab_size, self.embedding_size)
        self.lstm = torch.nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.output_prediction = torch.nn.Linear(self.hidden_size, self.output_vocab_size)

    def forward(self, input_index, h_c_state, encoder_outputs):
        """
        lookup the embedding for output vocabulary `input_index`,
        pass that embedding and the hidden state to an LSTM,
        predict the next token distribution based on the softmax of the output of the LSTM
        :param input_index:
        :param h_c_state:
        :param encoder_outputs:
        :return:
        """
        embedded_output = self.output_embeddings(input_index).view(1, 1, -1)
        output, h_c_state = self.lstm(embedded_output, h_c_state)

        softmax_input = self.output_prediction(output[0][0])

        output = torch.nn.functional.log_softmax(softmax_input, dim=0)
        return output, h_c_state

    def initial_h_c_state(self):
        return (torch.zeros(1, 1, self.hidden_size, device=DEVICE),
                torch.zeros(1, 1, self.hidden_size, device=DEVICE))


class TGenDec(BasicDecoder):
    def __init__(self, hidden_layer_size, output_vocab_size, max_input_length):
        super(TGenDec, self).__init__(hidden_layer_size, output_vocab_size, max_input_length)
        # Only define extra layers for attention here
        self.attention = torch.nn.Linear(self.hidden_size * 2, self.max_input_length)
        self.combining_attention = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input_index, h_c_state, encoder_outputs):
        embedded_output = self.output_embeddings(input_index).view(1, 1, -1)
        attention_input = torch.cat((embedded_output, h_c_state[0]), dim=2)
        attention_weights = torch.nn.functional.softmax(self.attention(attention_input), dim=2)
        attention_applied = torch.bmm(attention_weights, encoder_outputs)

        output = torch.cat((embedded_output[0], attention_applied[0]), 1)
        output = self.combining_attention(output).unsqueeze(0)

        # TGen uses tanh in its attention calculation
        output = torch.tanh(output)
        output, h_c_state = self.lstm(output, h_c_state)

        softmax_input = self.output_prediction(output[0][0])

        output = torch.nn.functional.log_softmax(softmax_input, dim=0)
        return output, h_c_state


class TGenEncDec(torch.nn.Module):
    def __init__(self,
                 input_vocab: "enunlg.vocabulary.IntegralInformVocabulary",
                 output_vocab: "enunlg.vocabulary.TokenVocabulary",
                 model_config=None):
        """
        TGenEncDec corresponds to TGen's seq2seq model with attention.

        TGen's model appears to be based on Tensorflow 0.6:
        https://github.com/tensorflow/tensorflow/blob/v0.6.0/tensorflow/python/ops/seq2seq.py

        :param input_vocab:
        :param output_vocab:
        :param model_config:
        """
        super().__init__()
        if model_config is None:
            # Set defaults
            model_config = omegaconf.DictConfig({'name': 'tgen',
                                                 'max_input_length': 30,
                                                 'encoder':
                                                     {'embeddings':
                                                          {'mode': 'random',
                                                           'dimensions': 50,
                                                           'backprop': True
                                                           },
                                                      'cell': 'lstm',
                                                      'num_hidden_dims': 50},
                                                 'decoder':
                                                     {'embeddings':
                                                          {'mode': 'random',
                                                           'dimensions': 50,
                                                           'backprop': True
                                                           },
                                                      'cell': 'lstm',
                                                      'num_hidden_dims': 50
                                                      }
                                                 })
        self.config = model_config

        # Set basic properties
        self.input_vocab = input_vocab
        self.input_vocab_size = input_vocab.max_index + 1
        self.output_vocab = output_vocab
        self.output_vocab_size = output_vocab.max_index + 1
        self.output_stop_token = self.output_vocab.stop_token_int

        # Initialize encoder and decoder networks
        # TODO either tie enc-dec num hidden dims together or add code to handle config when they have diff dimensionality
        self.encoder = TGenEnc(self.input_vocab_size, self.config.encoder.num_hidden_dims)
        self.decoder = TGenDec(self.config.decoder.num_hidden_dims, self.output_vocab_size, self.config.max_input_length)

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
        """
        Function to perform the computation that produces the output.

        This can perform arbitrary computation involving any number of inputs and outputs.
        """
        enc_outputs, enc_h_c_state = self.encode(enc_emb)

        dec_h_c_state = enc_h_c_state
        # TODO Replace this with a reference to the START symbol in the vocabulary
        dec_output = 1
        dec_outputs = [dec_output]

        for _ in range(max_output_length):
            dec_output, dec_h_c_state = self.decoder(dec_output, dec_h_c_state, enc_outputs)
            topv, topi = dec_output.data.topk(1)
            dec_outputs.append(topi.item())
            # TODO Replace this with a reference to the STOP symbol in the vocabulary
            # TODO actually, this should fill the rest of the outputs up to max_output_length with the VOID symbol in the vocab
            if topi.item() == 2:
                break
            dec_output = topi.squeeze().detach()
            dec_outputs.append(dec_output.unsqueeze(0))
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
        return loss.item() / dec_emb.size(0)

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
            enc_outputs, enc_h_c_state = self.encode(enc_emb)

            dec_input = torch.tensor([[1]], device=DEVICE)

            dec_h_c_state = enc_h_c_state

            dec_outputs = [1]

            for dec_index in range(max_length):
                dec_output, dec_h_c_state = self.decoder(dec_input, dec_h_c_state, enc_outputs)
                topv, topi = dec_output.data.topk(1)
                dec_outputs.append(topi.item())
                if topi.item() == 2:
                    break
                dec_input = topi.squeeze().detach()
            return dec_outputs

    def input_rep_to_string(self, input_token_indices) -> str:
        return self.input_vocab.pretty_string(input_token_indices)

    def output_rep_to_string(self, output_token_indices) -> str:
        return self.output_vocab.pretty_string(output_token_indices)
