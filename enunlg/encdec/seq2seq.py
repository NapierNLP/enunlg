from typing import List, Optional, Tuple

import logging

import omegaconf
import torch
import torch.nn
import torch.nn.functional

logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")


class BasicLSTMEncoder(torch.nn.Module):
    def __init__(self, num_unique_inputs, num_embedding_dims, num_hidden_dims, init="zeros"):
        """
        :param num_unique_inputs:
        :param num_embedding_dims:
        :param num_hidden_dims:
        """
        super(BasicLSTMEncoder, self).__init__()
        # Define properties
        self.num_unique_inputs = num_unique_inputs
        self.num_hidden_dims = num_hidden_dims
        self.num_embedding_dims = num_embedding_dims

        # Initialise embedding and LSTM
        self.embedding = torch.nn.Embedding(self.num_unique_inputs, self.num_embedding_dims)
        self.lstm = torch.nn.LSTM(self.num_embedding_dims, self.num_hidden_dims, batch_first=True)

        if init == "zeros":
            self._hidden_state_init_func = torch.zeros
        else:
            self._hidden_state_init_func = lambda *args: torch.randn(*args)/torch.sqrt(torch.Tensor([self.num_hidden_dims]))

    def forward(self, input_indices, h_c_state):
        # This assumes batch first and batch size = 1
        embedded = self.embedding(input_indices).view(1, len(input_indices), -1)
        return self.lstm(embedded, h_c_state)

    def forward_one_step(self, input_index, h_c_state):
        # This assumes batch first and batch size = 1
        embedded = self.embedding(input_index).view(1, 1, -1)
        output, h_c_state = self.lstm(embedded, h_c_state)
        return output, h_c_state

    def initial_h_c_state(self):
        # This assumes batch first and batch size = 1
        return (self._hidden_state_init_func(1, 1, self.num_hidden_dims, device=DEVICE),
                self._hidden_state_init_func(1, 1, self.num_hidden_dims, device=DEVICE))


class BasicDecoder(torch.nn.Module):
    def __init__(self, hidden_layer_size, output_vocab_size, max_input_length,
                 num_embedding_dims=None, padding_idx=None, start_token_idx=None, stop_token_idx=None):
        super(BasicDecoder, self).__init__()
        self.hidden_size = hidden_layer_size
        if num_embedding_dims is None:
            self.embedding_size = self.hidden_size
        else:
            self.embedding_size = num_embedding_dims
        self.padding_idx = padding_idx
        self.start_idx = start_token_idx
        self.stop_idx = stop_token_idx
        self.output_vocab_size = output_vocab_size
        self.max_input_length = max_input_length

        self.output_embeddings = torch.nn.Embedding(self.output_vocab_size, self.embedding_size, padding_idx=self.padding_idx)
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


class LSTMDecWithAttention(BasicDecoder):
    def __init__(self, hidden_layer_size, output_vocab_size, max_input_length,
                 num_embedding_dims=None, padding_idx=None, start_token_idx=None, stop_token_idx=None):
        super(LSTMDecWithAttention, self).__init__(hidden_layer_size, output_vocab_size, max_input_length,
                                                   num_embedding_dims=num_embedding_dims, padding_idx=padding_idx,
                                                   start_token_idx=start_token_idx, stop_token_idx=stop_token_idx)
        # Only define extra layers for attention here
        # Not sure exactly why the max input length needs to be two longer -- can't figure out where that's coming from
        self.attention = torch.nn.Linear(self.hidden_size + self.embedding_size, self.max_input_length)
        self.combining_attention = torch.nn.Linear(self.hidden_size + self.embedding_size, self.embedding_size)

    def forward(self, input_index, h_c_state, encoder_outputs):
        embedded_output = self.output_embeddings(input_index).view(1, 1, -1)
        logger.debug(f"{embedded_output.size()=}")
        attention_input = torch.cat((embedded_output, h_c_state[0]), dim=2)
        attention_weights = torch.nn.functional.softmax(self.attention(attention_input), dim=2)
        logger.debug(f"{attention_weights.size()=}")
        logger.debug(f"{encoder_outputs.size()=}")
        attention_applied = torch.bmm(attention_weights, encoder_outputs)

        output = torch.cat((embedded_output[0], attention_applied[0]), 1)
        output = self.combining_attention(output).unsqueeze(0)

        # TGen uses tanh in its attention calculation
        output = torch.tanh(output)
        output, h_c_state = self.lstm(output, h_c_state)

        softmax_input = self.output_prediction(output[0][0])

        output = torch.nn.functional.log_softmax(softmax_input, dim=0)
        return output, h_c_state


class Seq2SeqAttn(torch.nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, model_config=None):
        super().__init__()
        if model_config is None:
            # Set defaults
            model_config = omegaconf.DictConfig({'name': 'seq2seq+attn',
                                                 'max_input_length': 32,
                                                 'encoder':
                                                     {'embeddings':
                                                         {'type': 'torch',
                                                          'num_embeddings': input_vocab_size,
                                                          'embedding_dim': 64,
                                                          'backprop': True
                                                          },
                                                      'cell': 'lstm',
                                                      'num_hidden_dims': 128},
                                                 'decoder':
                                                     {'embeddings':
                                                         {'type': 'torch',
                                                          'num_embeddings': output_vocab_size,
                                                          'embedding_dim': 64,
                                                          'backprop': True
                                                         },
                                                      'cell': 'lstm',
                                                      'num_hidden_dims': 128
                                                     }
                                                 })
        if "num_embeddings" in model_config.encoder.embeddings:
            assert input_vocab_size == model_config.encoder.embeddings.num_embeddings
        if "num_embeddings" in model_config.decoder.embeddings:
            assert output_vocab_size == model_config.decoder.embeddings.num_embeddings
        self.config = model_config

        # Set basic properties
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        # Initialize encoder and decoder networks
        # TODO either tie enc-dec num hidden dims together or add code to handle config when they have diff dimensionality
        self.encoder = BasicLSTMEncoder(self.input_vocab_size,
                                        self.config.encoder.embeddings.embedding_dim,
                                        self.config.encoder.num_hidden_dims)
        self.decoder = LSTMDecWithAttention(self.config.decoder.num_hidden_dims,
                                            self.output_vocab_size,
                                            self.config.max_input_length,
                                            padding_idx=self.config.decoder.embeddings.get('padding_idx'),
                                            start_token_idx=self.config.decoder.embeddings.get('start_idx'),
                                            stop_token_idx=self.config.decoder.embeddings.get('stop_idx'))

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
        dec_input = torch.tensor([[self.decoder.start_idx]], device=DEVICE)
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
        # Use torch.zeros because we use padding_idx = 0
        dec_outputs = torch.zeros((len(dec_emb), self.output_vocab_size))
        # the first element of dec_emb is the start token
        dec_outputs[0] = dec_emb[0]
        # That's also why we skip the first element in our loop
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
