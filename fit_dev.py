import time
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data_dev import Seq2SeqDataManager, to_padded_tensor, normalize_string, TOK_XX
from masked_cross_entropy import *
from decoder import LuongAttnDecoderRNN
from encoder import EncoderRNN
from utils import print_summary

USE_CUDA = False
DEVICE='cpu'
#PAD_token = 1
#SOS_token = 1
#SOS_token = 2
#EOS_token = 2
#EOS_token = 3
MIN_LENGTH = 3
#MAX_LENGTH = 25
MAX_LENGTH = 6
MIN_COUNT = 3

## Get data
#data_manager=Seq2SeqDataManager.create_from_txt('data/eng-fra_sub.txt')
data_manager=Seq2SeqDataManager.create_from_txt('data/eng-fra_sub.txt', min_freq=MIN_COUNT, min_ntoks=MIN_LENGTH,
                                                max_ntoks=MAX_LENGTH, switch_pair=True, device=DEVICE)
#data_manager=Seq2SeqDataManager.create_from_txt('data/eng-fra.txt', min_ntoks=3, max_ntoks=10)
##test
train_batch_size = 100
valid_batch_size=100
train_dataloader, valid_dataloader=data_manager.get_dataloaders(train_batch_size=train_batch_size, valid_batch_size=valid_batch_size)


##model conf
# Configure models
attn_model = 'dot'
#hidden_size = 500
hidden_size = 50
n_layers = 2
dropout = 0.1
#batch_size = 100


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
#learning_rate = 0.0001
learning_rate = 0.001
decoder_learning_ratio = 5.0
# n_epochs = 50000
n_epochs =20
epoch = 0
# plot_every = 20
plot_every = 2000
# print_every = 100

# evaluate_every = 1000
evaluate_every = 1

# Initialize models
encoder = EncoderRNN(len(data_manager.train_seq2seq.seq_x.vocab.itos), hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, hidden_size, len(data_manager.train_seq2seq.seq_y.vocab.itos), n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
#decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

##train helper
def train_batch(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
                decoder_optimizer, batch_size, clip, MAX_LENGTH, loss_func=masked_cross_entropy, TOK_XX=TOK_XX, device='cpu'):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    #decoder_input = torch.LongTensor([TOK_XX.BOS_id] * batch_size)
    decoder_input = torch.tensor([TOK_XX.BOS_id] * batch_size, device=device)
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    #target_lengths = torch.tensor(target_lengths, device=device)
    max_target_length = max(target_lengths)

    #print(max_target_length)
    #max_target_length = MAX_LENGTH
    #max_target_length = 12
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size, device=device)

    # Move new Variables to CUDA
    #if USE_CUDA:
        #decoder_input = decoder_input.cuda()
        #all_decoder_outputs = all_decoder_outputs.cuda()

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
        target_lengths
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
                valid_batch_size, MAX_LENGTH, TOK_XX=TOK_XX, device='cpu'):
    encoder_outputs, encoder_hidden = encoder(val_input_batches, val_input_lengths, None)

    # Create starting vectors for decoder
    # decoder_input = torch.LongTensor([SOS_token])  # SOS
    #decoder_input = torch.LongTensor([TOK_XX.BOS_id] * valid_batch_size)  # SOS
    decoder_input = torch.tensor([TOK_XX.BOS_id] * valid_batch_size, device=device)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    #if USE_CUDA:
     #   decoder_input = decoder_input.cuda()

    # Run through decoder
    val_target_max_len = max(val_target_lengths)
    #val_target_lengths = torch.tensor(val_target_lengths, device=device)
    #val_target_max_len = MAX_LENGTH
    all_decoder_outputs = torch.zeros(val_target_max_len, valid_batch_size, decoder.output_size, device=device)

    # Store output words and attention states
    # decoded_words = []
    decoded_words = [[]] * valid_batch_size
    # decoder_attentions = torch.zeros(MAX_LENGTH + 1, MAX_LENGTH + 1)
    #decoder_attentions = torch.zeros(valid_batch_size, MAX_LENGTH + 1, MAX_LENGTH + 1)
    decoder_attentions = torch.zeros(valid_batch_size, val_target_max_len + 1, val_target_max_len + 1)

    for di in range(val_target_max_len):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs )
        all_decoder_outputs[di] = decoder_output
        decoder_attentions[:, di, :decoder_attention.size(2)] += decoder_attention.squeeze(1).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        for i in range(len(topv)):
            ni = topi[i][0]
            if ni.item() == TOK_XX.EOS_id:
                decoded_words[i].append(TOK_XX.EOS)
                break
            else:
                decoded_words[i].append(data_manager.train_seq2seq.seq_y.vocab.itos[ni.item()])  # another change
        # Next input is chosen word
        decoder_input = torch.tensor(topi.squeeze(), device=device)

        with torch.no_grad():
            loss = masked_cross_entropy(
                all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
                val_target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
                val_target_lengths
            )

    #return loss, decoder_attentions[:, di+1, :len(encoder_outputs)], decoded_words
    return loss, decoder_attentions, decoded_words

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def fit(epochs, encoder, encoder_optimizer, decoder, decoder_optimizer, batch_size,valid_batch_size, clip, MAX_LENGTH, loss_func=masked_cross_entropy,
        train_dl=None, valid_dl=None, TOK_XX=TOK_XX, device='cpu'):
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
                decoder_optimizer, train_batch_size, clip, MAX_LENGTH, loss_func, TOK_XX, device)

            nbatches_train+=1
            loss_total_train += loss
            eca += ec
            dca += dc
        train_loss_avg = loss_total_train / nbatches_train

        encoder.train(False)
        decoder.train(False)
        loss_total_valid = 0
        nbatches_valid = 0
        #valid_batch_size=batch_size*2 #no need to compute gradient, make batch bigger
        valid_batch_size_temp=valid_batch_size
        for val_input_batches, val_input_lengths, val_target_batches, val_target_lengths in valid_dl:
            if val_input_batches.size()[1] != valid_batch_size_temp:
                #continue
                valid_batch_size_temp=val_input_batches.size()[1]
            loss, decoder_attentions, decoded_words=valid_batch(encoder, decoder, val_input_batches, val_input_lengths,
                                                                val_target_batches, val_target_lengths,
                                                                valid_batch_size_temp, MAX_LENGTH, TOK_XX, device)
            nbatches_valid+=1
            loss_total_valid+=loss

        valid_loss_avg=loss_total_valid/nbatches_valid
        print_summary(start, epoch, epochs, train_loss_avg.item(), valid_loss_avg.item())
        #show_attention(data_manager.valid_seq2seq.seq_x.vocab.textify(val_input_batches[0]), decoded_words[0], decoder_attentions[0])


def predict(text, encoder, decoder, data_manager, max_length=10, device='cpu'):
    encoder.train(False)
    decoder.train(False)
    text=normalize_string(text)
    input_toks_id=data_manager.train_seq2seq.seq_x.numericalize(text)
    input_batch, input_length=to_padded_tensor([input_toks_id], device=device)

    #input_batch=input_batch.to(device)
    encoder_outputs, encoder_hidden = encoder(input_batch, input_length, None)
    decoder_input = torch.tensor([TOK_XX.BOS_id], device=device)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    decoded_toks_id = []
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        #decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni.item() == TOK_XX.EOS_id:
            decoded_toks_id.append(ni.item())
            break
        else:
            decoded_toks_id.append(ni.item())

        decoder_input = torch.tensor([ni], device=device)
        #if USE_CUDA:
            #decoder_input = decoder_input.cuda()
        #decoder_input = decoder_input.to(device)

    #turn them into words
    text=data_manager.train_seq2seq.seq_y.textify(decoded_toks_id)
    return text


#test
fit(n_epochs, encoder, encoder_optimizer, decoder, decoder_optimizer, train_batch_size, valid_batch_size, clip,MAX_LENGTH+1,
  masked_cross_entropy, train_dl=train_dataloader, valid_dl=valid_dataloader, device=DEVICE)


original_xtext='Je suis sûr.'
original_ytext='I am sure.'
predicted_text=predict(original_xtext, encoder, decoder, data_manager, device=DEVICE)
print(f'original text: {original_xtext}')
print(f'original answer: {original_ytext}')
print(f'predicted text: {predicted_text}')