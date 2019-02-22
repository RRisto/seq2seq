import torch.nn as nn
import torch.nn.functional as F
from seq2seq.utils.masked_cross_entropy import *
from seq2seq.model.attention import Attn


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, itos, output_size, max_length, n_layers=1, dropout_p=0.1, emb_vecs=None):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = len(itos)
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # Define layers
        self.embedding = self._init_embedding(itos, self.hidden_size, emb_vecs)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, output_size)


    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # TODO: FIX BATCHING

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        context = context.transpose(0, 1) # 1 x B x N

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

    def _init_embedding(self, itos, em_sz, emb_vecs=None):
        emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
        if emb_vecs is not None:
            wgts = emb.weight.data
            miss = []
            for i, w in enumerate(itos):
                try:
                    wgts[i] = torch.from_numpy(list(emb_vecs[w]))
                except:
                    miss.append(w)
            print(len(miss), miss[5:10])
        return emb


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, itos, hidden_size, n_layers=1, dropout=0.1, emb_vecs=None):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = len(itos)
        self.n_layers = n_layers
        self.dropout = dropout
        # Define layers
        self.embedding = self._init_embedding(itos, self.hidden_size, emb_vecs)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, self.hidden_size)


    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


    def _init_embedding(self, itos, em_sz, emb_vecs=None):
        emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
        if emb_vecs is not None:
            wgts = emb.weight.data
            miss = []
            for i, w in enumerate(itos):
                try:
                    wgts[i] = torch.from_numpy(list(emb_vecs[w]))
                except:
                    miss.append(w)
            print(len(miss), miss[5:10])
        return emb

