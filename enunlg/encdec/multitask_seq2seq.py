from pathlib import Path
from typing import List, Optional, Tuple

import logging
import tarfile

import omegaconf
import torch
import torch.nn
import torch.nn.functional

from enunlg.util import log_list_of_tensors_sizes

import enunlg.encdec.seq2seq as s2s

logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")


class MultitaskLSTMEncoder(s2s.BasicLSTMEncoder):
    def __init__(self, num_unique_inputs, num_embedding_dims, num_hidden_dims, task_names, init="zeros"):
        """
        :param num_unique_inputs:
        :param num_embedding_dims:
        :param num_hidden_dims:
        :param task_names:
        :param init:
        """
        super(MultitaskLSTMEncoder, self).__init__(num_unique_inputs, num_embedding_dims, num_hidden_dims, init)
        self.task_names = task_names
        self.num_tasks = len(task_names)

        # First layer is self.lstm
        self.layers = torch.nn.ModuleDict({self.task_names[0]: self.lstm})
        for task_name in task_names[1:]:
            self.layers[task_name] = torch.nn.LSTM(self.num_hidden_dims, self.num_hidden_dims, batch_first=True)

        if init == "zeros":
            self._hidden_state_init_func = torch.zeros
        else:
            self._hidden_state_init_func = lambda *args: torch.randn(*args)/torch.sqrt(torch.Tensor([self.num_hidden_dims]))

    def forward(self, input_indices: torch.Tensor, init_h_c_state):
        # This assumes batch first and batch size = 1
        embedded = self.embedding(input_indices).view(1, input_indices.size()[0], -1)
        outputs, h_c_state = [], []
        layer_outputs, layer_h_c_state = self.lstm(embedded, init_h_c_state)
        outputs.append(layer_outputs)
        h_c_state.append(layer_h_c_state)
        for task in self.task_names[1:]:
            layer_outputs, layer_h_c_state = self.layers[task](outputs[-1], h_c_state[-1])
            outputs.append(layer_outputs)
            h_c_state.append(layer_h_c_state)
        return outputs, h_c_state

    def forward_one_step(self, input_index, h_c_state):
        raise NotImplementedError()

    def initial_h_c_state(self):
        # This assumes batch first and batch size = 1
        return (self._hidden_state_init_func(1, 1, self.num_hidden_dims),
                self._hidden_state_init_func(1, 1, self.num_hidden_dims))


class DeepEncoderMultiDecoderSeq2SeqAttn(torch.nn.Module):
    def __init__(self, layer_names: List[str], layer_vocab_sizes: List[int], model_config: omegaconf.DictConfig):
        """
        We have len(layer_names) - 1 LSTM layers in the Encoder and the same number of tasks for our  decoder.
        The last layer is the output layer, and the first is the input, each intermediate layer is a different pipeline NLG task.

        :param layer_names: names for each of the annotation layers (in order from input to output)
        :param layer_vocab_sizes:
        :param model_config:
        """
        super().__init__()
        self.config = model_config

        # Set basic properties
        self.layer_names = layer_names
        self.layer_vocab_sizes = layer_vocab_sizes

        # Initialize encoder and decoder networks
        self.encoder = MultitaskLSTMEncoder(self.layer_vocab_sizes[0],
                                            self.config.encoder.embeddings.embedding_dim,
                                            self.config.encoder.num_hidden_dims,
                                            self.layer_names[1:])
        self.task_decoders = torch.nn.ModuleDict()
        for idx, name in enumerate(layer_names[1:]):
            self.task_decoders[name] = s2s.LSTMDecWithAttention(self.config[f"decoder_{name}"].num_hidden_dims,
                                                                self.layer_vocab_sizes[idx+1],
                                                                self.config.max_input_length,
                                                                self.config[f'decoder_{name}'].embeddings.embedding_dim,
                                                                padding_idx=self.config[f"decoder_{name}"].embeddings.get('padding_idx'),
                                                                start_token_idx=self.config[f"decoder_{name}"].embeddings.get('start_idx'),
                                                                stop_token_idx=self.config[f"decoder_{name}"].embeddings.get('stop_idx'))

        self._max_input_length = None

    @property
    def max_input_length(self):
        if self._max_input_length is None:
            self._max_input_length = self.task_decoders['raw_output'].get_parameter("attention.bias").size()[0]
        return self._max_input_length

    def encode(self, enc_emb: torch.Tensor):
        """
        Run encoding for a single set of integer inputs.

        :param enc_emb: tensor of integer inputs (shape?)
        :return: enc_outputs, enc_h_c_state (final state of the encoder)
        """
        enc_h_c_state = self.encoder.initial_h_c_state()
        # enc_outputs, enc_h_c_state
        return self.encoder(enc_emb, enc_h_c_state)

    def forward(self, enc_emb):
        return self.forward_e2e(enc_emb)

    def forward_multitask(self, enc_emb: torch.Tensor, dec_emb: torch.Tensor, teacher_forcing: float = 0.0, teacher_forcing_sync_layers: bool = True):
        """
        Do a forward pass generating from enc_emb using the targets in dec_emb as oracle values during decoding.
        If  teacher_forcing > 0, then dec_emb targets will be used as oracles only `teacher_forcing` proportion of the time.
        If teacher_forcing_sync_layers is True, then the teacher_forcing decision is made once for all layers (as opposed to being sampled for each layer).

        :param enc_emb:
        :param dec_emb:
        :param teacher_forcing:
        :param teacher_forcing_sync_layers:
        """
        # logger.debug(enc_emb.size())
        enc_outputs, enc_h_c_states = self.encode(enc_emb)
        # logger.debug(f"{len(enc_outputs)=}")
        # for x in enc_outputs:
        #     logger.debug(x.size())
        # # logger.debug(f"{len(enc_h_c_states)=}")
        # for x in enc_h_c_states:
        #     logger.debug(f"{x[0].size()=}, {x[1].size()=}")
        #
        # logger.debug(f"{len(dec_emb)=}")
        # for x in dec_emb:
        #     logger.debug(x.size())

        # it should be possible to parallelise the decoders for the different layers -- each one is autoregressive,
        # but they are independent of each other
        # We probs can't easily parallelise on the CPU, but we can save the function calls to enumerate and zip at least?
        outputs = []
        for idx, (layer_name, enc_output, enc_h_c_state, layer_dec_emb) in enumerate(zip(self.layer_names[1:], enc_outputs, enc_h_c_states, dec_emb), 1):
            dec_h_c_state = enc_h_c_state
            # Use torch.zeros because we use padding_idx = 0
            dec_outputs = torch.zeros((len(layer_dec_emb), self.layer_vocab_sizes[idx]))
            dec_outputs[0] = layer_dec_emb[0]
            for dec_input_index, dec_input in enumerate(layer_dec_emb[:-1]):
                dec_output, dec_h_c_state = self.task_decoders[layer_name](dec_input, dec_h_c_state, enc_output)
                dec_outputs[dec_input_index + 1] = dec_output
            outputs.append(dec_outputs)
        return outputs

    def forward_e2e(self, enc_emb: torch.Tensor, max_output_length: int = 100):
        """
        Do a forward pass generating from enc_emb, with output length up to `max_output_length` tokens.

        :param enc_emb:
        :param max_output_length: default is 100 based on Enriched E2E corpus having max layer length 97.
        """
        enc_outputs, enc_h_c_states = self.encode(enc_emb)

        enc_output = enc_outputs[-1]
        enc_h_c_state = enc_h_c_states[-1]
        dec_h_c_state = enc_h_c_state

        final_layer_name = self.layer_names[-1]
        final_layer_decoder = self.task_decoders[final_layer_name]

        dec_input = torch.tensor([[final_layer_decoder.start_idx]], device=DEVICE)

        dec_outputs = torch.zeros(max_output_length)
        for dec_index in range(max_output_length):
            dec_output, dec_h_c_state = final_layer_decoder(dec_input, dec_h_c_state, enc_output)
            topv, topi = dec_output.data.topk(1)
            dec_outputs[dec_index] = topi.item()
            if topi.item() == self.task_decoders[final_layer_name].stop_idx:
                break
            dec_input = topi.squeeze().detach()
        return dec_outputs

    def train_step(self, enc_emb: torch.Tensor, dec_emb: torch.Tensor, optimizer, criterion):
        optimizer.zero_grad()

        dec_outputs = self.forward_multitask(enc_emb, dec_emb)
        # log_list_of_tensors_sizes(dec_outputs)

        dec_targets = [torch.tensor([x.unsqueeze(0) for x in task]) for task in dec_emb]
        # log_list_of_tensors_sizes(dec_targets)

        loss = criterion(dec_outputs[0], dec_targets[0])
        for outputs, targets in zip(dec_outputs[1:], dec_targets[1:]):
            loss += criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        # mean loss per word returned in order for losses for sents of diff lengths to be comparable
        return loss.item() / sum([emb.size(0) for emb in dec_emb])

    def generate(self, enc_emb, max_length=100):
        return self.generate_greedy(enc_emb, max_length)

    def generate_beam(self, enc_emb, max_length=100, beam_size=10, num_expansions: Optional[int] = None):
        if num_expansions is None:
            num_expansions = beam_size
        with torch.no_grad():
            enc_outputs, enc_h_c_states = self.encode(enc_emb)

            enc_output = enc_outputs[-1]
            enc_h_c_state = enc_h_c_states[-1]
            dec_h_c_state = enc_h_c_state

            final_layer_name = self.layer_names[-1]
            final_layer_decoder = self.task_decoders[final_layer_name]

            prev_beam: List[Tuple[float, Tuple[int, ...], torch.Tensor]] = [(0.0, (torch.tensor([[final_layer_decoder.start_idx]]), ), dec_h_c_state)]

            stop_and_padding = (final_layer_decoder.stop_idx,
                                final_layer_decoder.padding_idx)

            for dec_index in range(max_length - 1):
                curr_beam = []
                for prev_beam_prob, prev_beam_item, prev_beam_hidden_state in prev_beam:
                    prev_item_index = prev_beam_item[-1]
                    if prev_item_index in stop_and_padding:
                        curr_beam.append((prev_beam_prob, prev_beam_item, prev_beam_hidden_state))
                    else:
                        dec_input = torch.tensor([[prev_item_index]])
                        dec_output, dec_h_c_state = final_layer_decoder(dec_input, prev_beam_hidden_state, enc_output)
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


    def generate_greedy(self, enc_emb, max_length=100):
        with torch.no_grad():
            return self.forward_e2e(enc_emb, max_length)

    def _save_classname_to_dir(self, directory_path):
        if not Path(directory_path).exists():
            Path(directory_path).mkdir()
        with (Path(directory_path) / "__class__.__name__").open('w') as class_file:
            class_file.write(self.__class__.__name__)

    def save(self, filepath, tgz=True):
        Path(filepath).mkdir
        self._save_classname_to_dir(filepath)
        with (Path(filepath) / "_state_dict.pt").open('wb') as state_file:
            torch.save(self.state_dict(), state_file)
        with (Path(filepath) / "model_config.yaml").open('w') as config_file:
            omegaconf.OmegaConf.save(self.config, config_file)
        with (Path(filepath) / "_init_args.yaml").open('w') as init_args_file:
            omegaconf.OmegaConf.save({'layer_names': self.layer_names,
                                      'layer_vocab_sizes': self.layer_vocab_sizes},
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
        new_model = cls(layer_names=init_args['layer_names'], layer_vocab_sizes=init_args['layer_vocab_sizes'], model_config=model_config)
        new_model.load_state_dict(state_dict)
        return new_model


class ShallowEncoderMultiDecoderSeq2SeqAttn(torch.nn.Module):
    def __init__(self, layer_names: List[str], layer_vocab_sizes: List[int], model_config: omegaconf.DictConfig):
        """
        We have len(layer_names) - 1 LSTM layers in the Encoder and the same number of tasks for our  decoder.
        The last layer is the output layer, and the first is the input, each intermediate layer is a different pipeline NLG task.

        :param layer_names: names for each of the annotation layers (in order from input to output)
        :param layer_vocab_sizes:
        :param model_config:
        """
        super().__init__()
        self.config = model_config

        # Set basic properties
        self.layer_names = layer_names
        self.layer_vocab_sizes = layer_vocab_sizes

        # Initialize encoder and decoder networks
        self.encoder = s2s.BasicLSTMEncoder(self.layer_vocab_sizes[0],
                                            self.config.encoder.embeddings.embedding_dim,
                                            self.config.encoder.num_hidden_dims)
        self.task_decoders = torch.nn.ModuleDict()
        for idx, name in enumerate(layer_names[1:]):
            self.task_decoders[name] = s2s.LSTMDecWithAttention(self.config[f"decoder_{name}"].num_hidden_dims,
                                                                self.layer_vocab_sizes[idx+1],
                                                                self.config.max_input_length,
                                                                self.config[f'decoder_{name}'].embeddings.embedding_dim,
                                                                padding_idx=self.config[f"decoder_{name}"].embeddings.get('padding_idx'),
                                                                start_token_idx=self.config[f"decoder_{name}"].embeddings.get('start_idx'),
                                                                stop_token_idx=self.config[f"decoder_{name}"].embeddings.get('stop_idx'))

    def encode(self, enc_emb: torch.Tensor):
        """
        Run encoding for a single set of integer inputs.

        :param enc_emb: tensor of integer inputs (shape?)
        :return: enc_outputs, enc_h_c_state (final state of the encoder)
        """
        enc_h_c_state = self.encoder.initial_h_c_state()
        # enc_outputs, enc_h_c_state
        return self.encoder(enc_emb, enc_h_c_state)

    def forward(self, enc_emb):
        return self.forward_e2e(enc_emb)

    def forward_multitask(self, enc_emb: torch.Tensor, dec_emb: torch.Tensor, teacher_forcing: float = 0.0, teacher_forcing_sync_layers: bool = True):
        """
        Do a forward pass generating from enc_emb using the targets in dec_emb as oracle values during decoding.
        If  teacher_forcing > 0, then dec_emb targets will be used as oracles only `teacher_forcing` proportion of the time.
        If teacher_forcing_sync_layers is True, then the teacher_forcing decision is made once for all layers (as opposed to being sampled for each layer).

        :param enc_emb:
        :param dec_emb:
        :param teacher_forcing:
        :param teacher_forcing_sync_layers:
        """
        logger.debug(enc_emb.size())
        enc_output, enc_h_c_state = self.encode(enc_emb)

        outputs = []
        for idx, (layer_name, layer_dec_emb) in enumerate(zip(self.layer_names[1:], dec_emb), 1):
            dec_h_c_state = enc_h_c_state
            # Use torch.zeros because we use padding_idx = 0
            dec_outputs = torch.zeros((len(layer_dec_emb), self.layer_vocab_sizes[idx]))
            dec_outputs[0] = layer_dec_emb[0]
            for dec_input_index, dec_input in enumerate(layer_dec_emb[:-1]):
                dec_output, dec_h_c_state = self.task_decoders[layer_name](dec_input, dec_h_c_state, enc_output)
                dec_outputs[dec_input_index + 1] = dec_output
            outputs.append(dec_outputs)
        return outputs

    def forward_e2e(self, enc_emb: torch.Tensor, max_output_length: int = 100):
        """
        Do a forward pass generating from enc_emb, with output length up to `max_output_length` tokens.

        :param enc_emb:
        :param max_output_length: default is 100 based on Enriched E2E corpus having max layer length 97.
        """
        enc_output, enc_h_c_state = self.encode(enc_emb)
        dec_h_c_state = enc_h_c_state

        final_layer_name = self.layer_names[-1]
        final_layer_decoder = self.task_decoders[final_layer_name]

        dec_input = torch.tensor([[final_layer_decoder.start_idx]], device=DEVICE)

        # TODO: pre-fill a tensor with zeros and then store results in that tensor (append is expensive)
        dec_outputs = []
        for dec_index in range(max_output_length):
            dec_output, dec_h_c_state = final_layer_decoder(dec_input, dec_h_c_state, enc_output)
            topv, topi = dec_output.data.topk(1)
            dec_outputs.append(topi.item())
            if topi.item() == self.task_decoders[final_layer_name].stop_idx:
                break
            dec_input = topi.squeeze().detach()
        return dec_outputs

    def train_step(self, enc_emb: torch.Tensor, dec_emb: torch.Tensor, optimizer, criterion):
        optimizer.zero_grad()

        dec_outputs = self.forward_multitask(enc_emb, dec_emb)
        dec_targets = [torch.tensor([x.unsqueeze(0) for x in task]) for task in dec_emb]

        loss = criterion(dec_outputs[0], dec_targets[0])
        for outputs, targets in zip(dec_outputs[1:], dec_targets[1:]):
            loss += criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        # mean loss per word returned in order for losses for sents of diff lengths to be comparable
        return loss.item() / sum([emb.size(0) for emb in dec_emb])

    def generate(self, enc_emb, max_length=100):
        """Only implementing greedy for now."""
        with torch.no_grad():
            return self.forward_e2e(enc_emb, max_length)
