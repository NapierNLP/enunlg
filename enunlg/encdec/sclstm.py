"""SCLSTM Cells and RNNs based on https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py"""

from collections import namedtuple
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import enunlg.embeddings
    import enunlg.embeddings.onehot

import random
import torch
import torch.jit
import torch.nn
import torch.nn.functional

import box

import enunlg.embeddings.glove

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
SCLSTMState = namedtuple('SCLSTMState', ['hx', 'cx', 'dx'])


class JitLSTMCell(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))

    @torch.jit.script_method
    def forward(self,
                input_tensor: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = state
        gates = (torch.mm(input_tensor, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * cell_gate)
        hy = out_gate * torch.tanh(cy)

        return hy, (hy, cy)


class JitLSTMLayer(torch.jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @torch.jit.script_method
    def forward(self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class SCLSTMCellVanilla(torch.nn.Module):
    def __init__(self, full_cued_da_size, input_size, hidden_size):
        """
        SCLSTM Cell based on the above custom LSTM implementations

        :param full_cued_da_size:
        :param input_size:
        :param hidden_size:
        """
        super(SCLSTMCellVanilla, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Gate weight matrices from input to hidden layer size
        self.weight_ih = torch.nn.Parameter(0.3 * (2 * torch.rand(4 * self.hidden_size, self.input_size) - 1))
        # Gate weight matrices from hidden layer size to hidden layer size
        self.weight_hh = torch.nn.Parameter(0.3 * (2 * torch.rand(4 * self.hidden_size, self.hidden_size) - 1))
        # Bias terms
        self.bias_ih = torch.nn.Parameter(0.3 * (2 * torch.rand(4 * self.hidden_size) - 1))
        self.bias_hh = torch.nn.Parameter(0.3 * (2 * torch.rand(4 * self.hidden_size) - 1))

        # SCLSTM-specific
        self.full_cued_da_size = full_cued_da_size
        self.alpha = 1.0
        self.weight_wr = torch.nn.Parameter(0.3 * (2 * torch.rand(self.full_cued_da_size, self.input_size) - 1))
        self.weight_hr = torch.nn.Parameter(0.3 * (2 * torch.rand(self.full_cued_da_size, self.hidden_size) - 1))
        self.weight_dc = torch.nn.Parameter(0.3 * (2 * torch.rand(self.hidden_size, self.full_cued_da_size) - 1))

def forward(self,
                input_token_embedding: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Based on equations 1-9 in Wen et al. 2015b.

        :param input_token_embedding: tensor representing a single token
        :param state: input hidden state, cell state, and cued_dialogue_act
        :return: hidden state, (hidden state, cell state, updated cued_dialogue_act)
        """
        hx, cx, dx = state
        # print(input_tensor.size())
        # print(self.weight_ih.size())
        gates = (torch.mm(input_token_embedding, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        # TODO This implementation is as the paper describes, but the code Shawn released includes the previous dx and another weight matrix as well
        # Second term below should be multiplied by a constant alpha, which we are implicitly setting to 1 here.
        read_gate = torch.mm(input_token_embedding, self.weight_wr.t()) + torch.mm(hx, self.weight_hr.t())

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)
        read_gate = torch.sigmoid(read_gate)

        dy = (read_gate * dx)
        cy = (forget_gate * cx) + (in_gate * cell_gate) + torch.tanh(torch.mm(dy, self.weight_dc.t()))
        hy = out_gate * torch.tanh(cy)

        return hy, (hy, cy, dy)


class SCLSTMCellPaper(torch.nn.Module):
    def __init__(self, full_cued_da_size, input_size, hidden_size):
        """
        SCLSTM Cell as described in the [Wen et al. 2015b paper](http://www.aclweb.org/anthology/D15-1199)

        :param full_cued_da_size:
        :param input_size:
        :param hidden_size:
        """
        super(SCLSTMCellPaper, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Gate weight matrices from input to hidden layer size
        self.weight_ih = torch.nn.Parameter(0.3 * (2 * torch.rand(4 * self.hidden_size, self.input_size) - 1))
        # Gate weight matrices from hidden layer size to hidden layer size
        self.weight_hh = torch.nn.Parameter(0.3 * (2 * torch.rand(4 * self.hidden_size, self.hidden_size) - 1))

        # SCLSTM-specific
        self.full_cued_da_size = full_cued_da_size
        self.alpha = 1.0
        self.weight_wr = torch.nn.Parameter(0.3 * (2 * torch.rand(self.full_cued_da_size, self.input_size) - 1))
        self.weight_hr = torch.nn.Parameter(0.3 * (2 * torch.rand(self.full_cued_da_size, self.hidden_size) - 1))
        self.weight_dc = torch.nn.Parameter(0.3 * (2 * torch.rand(self.hidden_size, self.full_cued_da_size) - 1))

    def forward(self,
                input_token_embedding: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Based on equations 1-9 in Wen et al. 2015b.

        :param input_token_embedding: tensor representing a single token
        :param state: input hidden state, cell state, and cued_dialogue_act
        :return: hidden state, (hidden state, cell state, updated cued_dialogue_act)
        """
        hx, cx, dx = state
        gates = (torch.mm(input_token_embedding, self.weight_ih.t()) +
                 torch.mm(hx, self.weight_hh.t()))
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        # Second term below should be multiplied by a constant alpha, which we are implicitly setting to 1 here.
        read_gate = torch.mm(input_token_embedding, self.weight_wr.t()) + self.alpha * torch.mm(hx, self.weight_hr.t())

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)
        read_gate = torch.sigmoid(read_gate)

        dy = (read_gate * dx)
        cy = (forget_gate * cx) + (in_gate * cell_gate) + torch.tanh(torch.mm(dy, self.weight_dc.t()))
        hy = out_gate * torch.tanh(cy)

        return hy, (hy, cy, dy)


class SCLSTMCellReleased(torch.nn.Module):
    def __init__(self, dialogue_act_size, slot_value_size, input_size, hidden_size):
        """
        SCLSTM Cell as released in the [RNNLG code](https://github.com/shawnwun/RNNLG)

        :param dialogue_act_size: number of dimensions in the bitvectors representing dialogue acts
        :param slot_value_size: number of dimensions in the bitvectors representing slot-value pairs
        :param input_size: number of dimensions used for token embeddings
        :param hidden_size: number of dimensions for the hidden layers
        """
        super(SCLSTMCellReleased, self).__init__()
        self.act_size = dialogue_act_size
        self.slot_value_size = slot_value_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(0.3 * (2 * torch.rand(4 * self.hidden_size, self.input_size) - 1))
        self.weight_hh = torch.nn.Parameter(0.3 * (2 * torch.rand(4 * self.hidden_size, self.hidden_size) - 1))
        self.weight_svh = torch.nn.Parameter(0.3 * (2 * torch.rand(4 * self.hidden_size, self.slot_value_size) - 1))

        self.alpha = 1.0
        self.weight_wr = torch.nn.Parameter(0.3 * (2 * torch.rand(self.slot_value_size, self.input_size) - 1))
        self.weight_hr = torch.nn.Parameter(0.3 * (2 * torch.rand(self.slot_value_size, self.hidden_size) - 1))
        self.weight_svr = torch.nn.Parameter(0.3 * (2 * torch.rand(self.slot_value_size, self.slot_value_size) - 1))
        self.weight_dc = torch.nn.Parameter(0.3 * (2 * torch.rand(self.hidden_size, self.act_size + self.slot_value_size) - 1))

    # @torch.jit.script_method
    def forward(self,
                input_tensor: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        hx, cx, dx = state
        a0 = dx[:self.act_size]
        a0 = a0.unsqueeze(0)
        svx = dx[self.act_size:]
        svx = svx.unsqueeze(0)

        gates = (torch.mm(input_tensor, self.weight_ih.t()) +
                 torch.mm(hx, self.weight_hh.t()) +
                 torch.mm(svx, self.weight_svh.t()))
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        read_gate = torch.mm(input_tensor, self.weight_wr.t()) + self.alpha * torch.mm(hx,
                                                                                      self.weight_hr.t()) + torch.mm(
            svx, self.weight_svr.t())

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)
        read_gate = torch.sigmoid(read_gate)

        svy = (read_gate * svx)
        cy = (forget_gate * cx) + (in_gate * cell_gate) + torch.tanh(torch.mm(torch.cat((a0, svy), dim=1), self.weight_dc.t()))
        hy = out_gate * torch.tanh(cy)

        a0 = a0.squeeze(0)
        svy = svy.squeeze(0)

        return hy, (hy, cy, torch.cat((a0, svy)))


# class SCLSTMLayer(torch.jit.ScriptModule):
class SCLSTMLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args):
        """
        Generic SCLSTM Layer which can use different SCLSTM Cell types

        :param cell: the SCLSTMCell class to use
        :param cell_args: initialisation args for the chosen SCLSTMCell class
        """
        super(SCLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    # @torch.jit.script_method
    def forward(self,
                input: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        inputs = input.unbind(0)
        # outputs = torch.jit.annotate(List[torch.Tensor], [])
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


SCLSTM_DESCRIBED_CONFIG = box.Box({"name": "sclstm",
                                   "mr_size": 24,  # based on Wen et al. SFX Restaurant data
                                   "embeddings":
                                       {"mode": "random",  # could also be one-hot, word2vec, glove, zero, etc
                                        "dimensions": 80,
                                        "backprop": True
                                        },
                                   "num_hidden_dims": 80
                                   })
SCLSTM_RELEASED_CONFIG = SCLSTM_DESCRIBED_CONFIG
del SCLSTM_RELEASED_CONFIG.mr_size
# based on Wen et al. SFX Restaurant data, as above
SCLSTM_RELEASED_CONFIG.act_size = 9
SCLSTM_RELEASED_CONFIG.slot_value_size = 15


class BaseSCLSTMModel(torch.nn.Module):
    def __init__(self,
                 da_embedder: "enunlg.embeddings.onehot.DialogueActEmbeddings",
                 token_int_mapper: "enunlg.vocabulary.TokenVocabulary",
                 model_config=None,
                 sclstm_layer=None):
        """
        :param da_embedder:
        :param token_int_mapper:
        :param model_config:
        :param sclstm_layer:
        """
        super().__init__()
        self.config = model_config

        # Constants for equation 13 from Wen et al. 2015b.
        self.one_slot_per_word_eta = 0.0001
        self.one_slot_per_word_xi = 100

        # Set basic properties
        self.input_vocab = da_embedder
        self.input_vocab_size = da_embedder.dimensionality

        self.output_vocab = token_int_mapper
        self.output_vocab_size = token_int_mapper.max_index + 1
        self.output_stop_token = self.output_vocab.stop_token_int

        # Initialize networks
        self.token_embeddings = torch.nn.Embedding(self.output_vocab_size, self.config.embeddings.dimensions)
        if sclstm_layer is None:
            raise ValueError("Cannot initialise an SCLSTMModel without first defining the SCLSTM Layer")
        self.sclstm_layer = sclstm_layer
        self.output_prediction = torch.nn.Linear(self.config.num_hidden_dims, self.output_vocab_size)

        # Adding this to clip gradients to 1.0 in place, based on line 73 of nn_generator.py in Shawn's code
        # This may not be appropriate, depending on what exactly the following line of theano means:
        # gradients = T.grad(clip_gradient(self.cost, 1), self.params)
        # Here we might add this instead:
        # torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)

    def init_h_c_state(self, batch_size=1):
        return LSTMState(torch.zeros((batch_size, self.config.num_hidden_dims)),
                         torch.zeros((batch_size, self.config.num_hidden_dims)))

    def forward_no_forcing(self, cued_da_bitvector, prev_token, hcd_state=None):
        # print(f"Inside {self.__class__}.forward_no_forcing()")
        token_embeddings = self.token_embeddings(torch.tensor([prev_token]))
        # Unsqueeze to make it a single batch-first batch
        # Permute to get it into batch-second representation
        if hcd_state is None:
            initial_hidden = self.init_h_c_state()
            hcd_state = (initial_hidden[0], initial_hidden[1], cued_da_bitvector)
        token_embeddings = token_embeddings.unsqueeze(0).permute(1, 0, 2)
        output, hcd_state = self.sclstm_layer(token_embeddings, hcd_state)
        # print(f"{output.size()=}")
        softmax_input = self.output_prediction(output)
        # print(f"{softmax_input.size()=}")
        softmax_output = torch.nn.functional.log_softmax(softmax_input, dim=2)
        # print(f"{softmax_output.size()=}")
        return softmax_output, hcd_state

    def forward_with_teacher_forcing(self, cued_da_bitvector, token_sequence):
        # print(f"Inside {self.__class__}.forward_with_teacher_forcing()")
        # TODO in Shawn's code, they treat the dialogue _act_ and the slot-value pairs differently.
        # - The act gets passed along, but the slot-value pairs get affected by the read_gate
        token_embeddings = self.token_embeddings(token_sequence)
        # Unsqueeze to make it a single batch-first batch
        # Permute to get it into batch-second representation
        token_embeddings = token_embeddings.unsqueeze(0).permute(1, 0, 2)
        initial_hidden = self.init_h_c_state()
        output, hcd_state = self.sclstm_layer(token_embeddings, (initial_hidden[0], initial_hidden[1], cued_da_bitvector))
        # print(f"{output.size()=}")
        softmax_input = self.output_prediction(output)
        # print(f"{softmax_input.size()=}")
        softmax_output = torch.nn.functional.log_softmax(softmax_input, dim=2)
        # print(f"{softmax_output.size()=}")
        return softmax_output, hcd_state

    def one_at_a_time_costs(self, hcd_states):
        da_states = [da for _, _, da in hcd_states]
        cost = 0.0
        for da_1, da_2 in zip(da_states, da_states[1:]):
            cost += self.one_slot_per_word_eta * self.one_slot_per_word_xi ** torch.abs(da_1 - da_2)
        return cost

    def train_step(self,
                   cued_da_bitvector: torch.Tensor,
                   target_token_indices: torch.Tensor,
                   optimizer,
                   criterion,
                   prob_teacher_forcing=1.0):
        # print(f"Inside {self.__class__}.train_step()")
        optimizer.zero_grad()

        hcd_states = []
        if random.random() < prob_teacher_forcing:
            dec_outputs, last_hcd_state = self.forward_with_teacher_forcing(cued_da_bitvector, target_token_indices[:-1])
        else:
            dec_outputs = []
            for token in target_token_indices[:-1]:
                dec_output, last_hcd_state = self.forward_no_forcing(cued_da_bitvector, token)
                dec_outputs.append(dec_output)
                hcd_states.append(last_hcd_state)
            dec_outputs = torch.stack(dec_outputs).squeeze(1)
        # Squeezing the batch out of outputs
        loss = criterion(dec_outputs.squeeze(1), target_token_indices[1:]) +\
               torch.sum(torch.abs(last_hcd_state[-1]))

        if hcd_states:
            loss += self.one_at_a_time_cost(hcd_states)
        # print(f"{dec_outputs.data.topk(1)=}")
        # print(f"{dec_outputs.size()=}")
        # print(f"{dec_targets=}")
        # print(f"{dec_targets.size()=}")
        # print(f"{loss=}")
        # print("----")
        loss.backward()
        optimizer.step()
        return loss.item()

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
            return [(prob / len(seq), seq) for prob, seq, _ in prev_beam]

    def generate_greedy(self, cued_da_bitvector: torch.Tensor, max_length=50):
        # print(f"Inside {self.__class__}.generate_greedy()")
        with torch.no_grad():
            # Start with the <go> token
            prev_token = torch.tensor([1])
            # Include the start token in the output
            dec_outputs = [1]
            # Initialize the hidden state
            initial_hidden = self.init_h_c_state()
            # Include the DA bitvector in the SCLSTM state
            hcd_state = (initial_hidden[0], initial_hidden[1], cued_da_bitvector)
            for dec_index in range(max_length):
                dec_output, hcd_state = self.forward_no_forcing(cued_da_bitvector, prev_token, hcd_state)
                # # print(f"{prev_token.size()=}")
                # input_embedding = self.token_embeddings(prev_token).unsqueeze(0).permute(1, 0, 2)
                # # print(f"{input_embedding.size()=}")
                # # print(f"{hcd_state=}")
                # output, hcd_state = self.sclstm_layer(input_embedding, hcd_state)
                # # print(f"Updated {hcd_state=}")
                # # print(f"{output=}")
                # softmax_input = self.output_prediction(output)
                # # print(f"{softmax_input.size()=}")
                # softmax_output = torch.nn.functional.log_softmax(softmax_input, dim=2)
                # # print(f"{softmax_output.size()=}")
                topv, topi = dec_output.data.topk(1)
                # print(f"{softmax_output=}")
                # print(f"{topv=}")
                # print(f"{topi=}")
                dec_outputs.append(topi.item())
                # print(f"{dec_outputs=}")
                if topi.item() == 2:
                    break
                prev_token = torch.tensor([dec_outputs[-1]])
                # print('----')
            return dec_outputs

    def input_rep_to_string(self, input_bitvector) -> str:
        # TODO Implement this!
        return ""

    def output_rep_to_string(self, output_token_indices) -> str:
        return self.output_vocab.pretty_string(output_token_indices)


class SCLSTMModelAsDescribed(BaseSCLSTMModel):
    def __init__(self, da_embedder: "enunlg.embeddings.onehot.DialogueActEmbeddings",
                 token_int_mapper: "enunlg.vocabulary.TokenVocabulary", model_config=None):
        if model_config is None:
            # Set defaults
            model_config = SCLSTM_DESCRIBED_CONFIG
        sclstm_layer = SCLSTMLayer(SCLSTMCellPaper, model_config.mr_size, model_config.embeddings.dimensions,
                                   model_config.num_hidden_dims)
        super().__init__(da_embedder, token_int_mapper, model_config, sclstm_layer)
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)


class SCLSTMModelAsReleased(BaseSCLSTMModel):
    def __init__(self, da_embedder: "enunlg.embeddings.onehot.DialogueActEmbeddings",
                 token_int_mapper: "enunlg.vocabulary.TokenVocabulary", model_config=None):
        if model_config is None:
            # Set defaults
            model_config = SCLSTM_RELEASED_CONFIG
        sclstm_layer = SCLSTMLayer(SCLSTMCellReleased, model_config.act_size, model_config.slot_value_size,
                                   model_config.embeddings.dimensions,
                                   model_config.num_hidden_dims)
        super().__init__(da_embedder, token_int_mapper, model_config, sclstm_layer)
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)


class SCLSTMModelAsDescribedWithGlove(SCLSTMModelAsDescribed):
    def __init__(self, da_embedder: "enunlg.embeddings.onehot.DialogueActEmbeddings",
                 glove_filepath: str, model_config=None):
        token_int_mapper, embedding_layer = enunlg.embeddings.glove.GloVeEmbeddings.from_word_embedding_txt(
            glove_filepath, with_vocab=True)
        if model_config is None:
            model_config = SCLSTM_DESCRIBED_CONFIG
        model_config.embeddings.mode = 'glove'
        super().__init__(da_embedder, token_int_mapper, model_config=model_config)
        self.token_embeddings = embedding_layer
        self.token_embeddings.requires_grad_(model_config.embeddings.backprop)
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)


class SCLSTMModelAsReleasedWithGlove(SCLSTMModelAsReleased):
    def __init__(self, da_embedder: "enunlg.embeddings.onehot.DialogueActEmbeddings",
                 glove_filepath: str, model_config=None):
        token_int_mapper, embedding_layer = enunlg.embeddings.glove.GloVeEmbeddings.from_word_embedding_txt(
            glove_filepath, with_vocab=True)
        if model_config is None:
            model_config = SCLSTM_RELEASED_CONFIG
        model_config.embeddings.mode = 'glove'
        super().__init__(da_embedder, token_int_mapper, model_config=model_config)
        self.token_embeddings = embedding_layer
        self.token_embeddings.requires_grad_(model_config.embeddings.backprop)
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)


def test_script_lstm_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    state = LSTMState(torch.randn(batch, hidden_size),
                      torch.randn(batch, hidden_size))
    rnn = JitLSTMLayer(JitLSTMCell, input_size, hidden_size)
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    lstm = torch.nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_lstm_layer_custom_input(seq_len, batch_size, input_size, hidden_size):
    inputs = generate_test_sequence(seq_len, batch_size, input_size)
    print(inputs)
    print(inputs.size())
    state = LSTMState(torch.randn(batch_size, hidden_size),
                      torch.randn(batch_size, hidden_size))
    rnn = JitLSTMLayer(JitLSTMCell, input_size, hidden_size)
    out, out_state = rnn(inputs, state)

    # Control: pytorch native LSTM
    lstm = torch.nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inputs, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def generate_test_sequence(seq_len, batch_size, input_size):
    input_seqs = []
    input_types = set()
    for _ in range(batch_size):
        input_seqs.append(random.choices(list(range(10)), k=seq_len))
        input_types.update(input_seqs[-1])
    input_types = list(input_types)
    input_embeddings = torch.randn(len(input_types), input_size)
    # NB: batch first for easier iteration since that's how we build the input sequences above
    test_seq = torch.zeros((batch_size, seq_len, input_size))
    for batch_idx in range(batch_size):
        for inp_idx, inp in enumerate(input_seqs[batch_idx]):
            idx = input_types.index(inp)
            emb = input_embeddings[idx]
            test_seq[batch_idx][inp_idx] = emb
    # Switch back to batch second when returning the sequence
    return test_seq.permute(1, 0, 2)


if __name__ == "__main__":
    # Use original Pytorch benchmark test to check our implementation works
    test_script_lstm_layer(5, 2, 3, 7)
    # Use a similar test to confirm that our generate_test_sequence() function works
    test_script_lstm_layer_custom_input(seq_len=10, batch_size=3, input_size=7, hidden_size=13)

    seq_len, batch, input_size, hidden_size = 5, 2, 3, 7
    inp = torch.randn(seq_len, batch, input_size)
    mr_size = 13
    mr = torch.ones(13)
    state = SCLSTMState(torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size),
                        mr)
    rnn = SCLSTMLayer(SCLSTMCellVanilla, mr_size, input_size, hidden_size)
    out, out_state = rnn(inp, state)

    print(out.size())
    for elem in out_state:
        print(elem.size())
