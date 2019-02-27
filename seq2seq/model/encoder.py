import torch.nn as nn
from seq2seq.utils.masked_cross_entropy import *


class EncoderRNN(nn.Module):
    def __init__(self, itos:list, hidden_size:int, n_layers:int=1, dropout:float=0.1, emb_vecs:dict=None):
        super(EncoderRNN, self).__init__()

        self.input_size = len(itos)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = self._init_embedding(itos, self.hidden_size, emb_vecs)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)

    def _init_embedding(self, itos:list, em_sz:int, emb_vecs:dict=None):
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

    def forward(self, input_seqs:torch.tensor, input_lengths:torch.tensor, hidden:torch.tensor=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden
