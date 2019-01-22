import time
import torch.nn as nn
from torch import optim

from dev import Seq2SeqDataManager
from masked_cross_entropy import *
from data_helpers import prepare_data, random_batch, read_langs
from decoder import LuongAttnDecoderRNN
from encoder import EncoderRNN
from eval_helpers import evaluate_randomly
from train_helpers import train
from utils import time_since

USE_CUDA = False
PAD_token = 0
SOS_token = 1
EOS_token = 2
MIN_LENGTH = 3
MAX_LENGTH = 25
MIN_COUNT = 5


## Get data
data_manager=Seq2SeqDataManager.create_from_txt('data/eng-fra_sub.txt')
##test
bs = 10
trn_dataloader, valid_dataloader=data_manager.get_dataloaders(batch_size=bs)


##model conf
# Configure models
attn_model = 'dot'
#hidden_size = 500
hidden_size = 50
n_layers = 2
dropout = 0.1
#batch_size = 100
batch_size = 10

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
# n_epochs = 50000
n_epochs = 5
epoch = 0
# plot_every = 20
plot_every = 2000
# print_every = 100

# evaluate_every = 1000
evaluate_every = 1

# Initialize models
encoder = EncoderRNN(len(data_manager.seq_x.vocab.itos), hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, hidden_size, len(data_manager.seq_y.vocab.itos), USE_CUDA, n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()


def fit(epochs, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion, loss_func=None, opt=None, train_dl=None, valid_dl=None, print_every = 1):
    start = time.time()
    loss_total=0
    eca = 0
    dca = 0

    for epoch in range(epochs):
        for input_batches, input_lengths, target_batches, target_lengths in train_dl:
            # my dirty quick fix, last batch usualyy not full size this avoids error
            if input_batches.size()[1] != batch_size:
                continue

            # Run the train function
            loss, ec, dc = train(
                input_batches, input_lengths, target_batches, target_lengths,
                encoder, decoder, encoder_optimizer, decoder_optimizer,
                criterion, MAX_LENGTH, batch_size, clip, SOS_token, USE_CUDA)

            # Keep track of loss
            loss_total += loss
            eca += ec
            dca += dc

        loss_avg = loss_total / print_every
        loss_total = 0
        print_summary = f'{time_since(start, epoch+1 / epochs)} ({epoch} {epoch+1 / epochs * 100}%) {round(loss_avg.item(), 2)}'
        print(print_summary)

#test
fit(2, encoder,encoder_optimizer, decoder, decoder_optimizer, criterion, train_dl=trn_dataloader)