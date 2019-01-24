import time
import torch.nn as nn
from torch import optim

from dev import Seq2SeqDataManager
from masked_cross_entropy import *
from decoder import LuongAttnDecoderRNN
from encoder import EncoderRNN
from utils import print_summary

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
train_dataloader, valid_dataloader=data_manager.get_dataloaders(batch_size=bs)


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

##train helper
def train_batch(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
                decoder_optimizer, batch_size, clip, SOS_token, USE_CUDA, loss_func=masked_cross_entropy):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([SOS_token] * batch_size)
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = loss_func(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths, USE_CUDA
    )
    loss.backward()

    # Clip gradient norms
    ec=torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc=torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data, ec, dc


def valid_batch(encoder, decoder, val_input_batches, val_input_lengths, val_target_batches, val_target_lengths,
          batch_size, MAX_LENGTH, SOS_token, USE_CUDA):
    encoder_outputs, encoder_hidden = encoder(val_input_batches, val_input_lengths, None)

    # Create starting vectors for decoder
    # decoder_input = torch.LongTensor([SOS_token])  # SOS
    decoder_input = torch.LongTensor([SOS_token] * batch_size)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    # decoded_words = []
    decoded_words = [[]] * batch_size
    # decoder_attentions = torch.zeros(MAX_LENGTH + 1, MAX_LENGTH + 1)
    decoder_attentions = torch.zeros(batch_size, MAX_LENGTH + 1, MAX_LENGTH + 1)

    # Run through decoder
    val_target_max_len = max(val_target_lengths)
    all_decoder_outputs = torch.zeros(val_target_max_len, batch_size, decoder.output_size)

    for di in range(val_target_max_len):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs )
        all_decoder_outputs[di] = decoder_output
        decoder_attentions[:, di, :decoder_attention.size(2)] += decoder_attention.squeeze(1).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        for i in range(len(topv)):
            ni = topi[i][0]
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                # break
            else:
                decoded_words[i].append(data_manager.seq_y.vocab.itos[ni.item()])  # another change
        # Next input is chosen word
        decoder_input = torch.LongTensor(topi.squeeze())
        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        with torch.no_grad():
            loss = masked_cross_entropy(
                all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
                val_target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
                val_target_lengths, USE_CUDA
            )

    return loss


def fit(epochs, encoder, encoder_optimizer, decoder, decoder_optimizer, batch_size, clip, loss_func=masked_cross_entropy,
        train_dl=None, valid_dl=None):
    eca = 0
    dca = 0

    for epoch in range(epochs):
        encoder.train(True)
        decoder.train(True)

        start = time.time()
        loss_total_train = 0
        nbatches_train=0
        train_batch_size=batch_size
        for input_batches, input_lengths, target_batches, target_lengths in train_dl:
            # my dirty quick fix, last batch usually not full size this avoids error
            if input_batches.size()[1] != batch_size:
                #continue
                train_batch_size=input_batches.size()[1]

            loss, ec, dc = train_batch(
                input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
                decoder_optimizer, train_batch_size, clip, SOS_token, USE_CUDA, loss_func)

            nbatches_train+=1
            loss_total_train += loss
            eca += ec
            dca += dc
        train_loss_avg = loss_total_train / nbatches_train

        encoder.train(False)
        decoder.train(False)
        loss_total_valid = 0
        nbatches_valid = 0
        valid_batch_size=batch_size*2 #no need to compute gradient, make batch bigger
        for val_input_batches, val_input_lengths, val_target_batches, val_target_lengths in valid_dl:
            if val_input_batches.size()[1] != valid_batch_size:
                #continue
                valid_batch_size=val_input_batches.size()[1]
            loss=valid_batch(encoder, decoder, val_input_batches, val_input_lengths, val_target_batches, val_target_lengths,
                        valid_batch_size, MAX_LENGTH, SOS_token, USE_CUDA)
            nbatches_valid+=1
            loss_total_valid+=loss

        valid_loss_avg=loss_total_valid/nbatches_valid
        print_summary(start, epoch, epochs, train_loss_avg.item(), valid_loss_avg.item())


def predict(text, encoder, decoder):
    encoder.train(False)
    decoder.train(False)


#test
fit(2, encoder, encoder_optimizer, decoder, decoder_optimizer, batch_size, clip, masked_cross_entropy, train_dl=train_dataloader, valid_dl=valid_dataloader)