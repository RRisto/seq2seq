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


## Data
#input_lang, output_lang, pairs = prepare_data('eng', 'fra',MIN_LENGTH, MAX_LENGTH, True)


data_manager=Seq2SeqDataManager.create_from_txt('data/eng-fra_sub.txt')
##test
small_batch_size = 10
trn_dataloader, valid_dataloader=data_manager.get_dataloaders(batch_size=small_batch_size)

#input_batches, input_lengths, target_batches, target_lengths = random_batch(small_batch_size, pairs, input_lang,
#                                                                            output_lang, PAD_token, EOS_token, USE_CUDA)
input_batch, input_lens, target_batch, target_lens=next(iter(trn_dataloader))
print('input_batches', input_batch.size()) # (max_len x batch_size)
print('target_batches', target_batch.size()) # (max_len x batch_size)

small_hidden_size = 8
small_n_layers = 2

encoder_test = EncoderRNN(len(data_manager.seq_x.vocab.itos), small_hidden_size, small_n_layers)
decoder_test = LuongAttnDecoderRNN('general', small_hidden_size, len(data_manager.seq_y.vocab.itos), USE_CUDA, small_n_layers)

encoder_outputs, encoder_hidden = encoder_test(input_batch, input_lens, None)

print('encoder_outputs', encoder_outputs.size()) # max_len x batch_size x hidden_size
print('encoder_hidden', encoder_hidden.size()) # n_layers * 2 x batch_size x hidden_size

max_target_length = max(target_lens)

# Prepare decoder input and outputs
decoder_input = torch.LongTensor([SOS_token] * small_batch_size)
decoder_hidden = encoder_hidden[:decoder_test.n_layers] # Use last (forward) hidden state from encoder
all_decoder_outputs = torch.zeros(max_target_length, small_batch_size, decoder_test.output_size)

if USE_CUDA:
    all_decoder_outputs = all_decoder_outputs.cuda()
    decoder_input = decoder_input.cuda()

# Run through decoder one time step at a time
for t in range(max_target_length):
    decoder_output, decoder_hidden, decoder_attn = decoder_test(
        decoder_input, decoder_hidden, encoder_outputs
    )
    all_decoder_outputs[t] = decoder_output # Store this step's outputs
    decoder_input = target_batch[t] # Next input is current target

# Test masked cross entropy loss
loss = masked_cross_entropy(
    all_decoder_outputs.transpose(0, 1).contiguous(),
    target_batch.transpose(0, 1).contiguous(),
    target_lens,
    USE_CUDA
)
print('loss', loss.data)

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
n_epochs = 2
epoch = 0
# plot_every = 20
plot_every = 2000
# print_every = 100
print_every = 500
# evaluate_every = 1000
evaluate_every = 300

# Initialize models
encoder = EncoderRNN(len(data_manager.seq_x.vocab.itos), hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, hidden_size, len(data_manager.seq_y.vocab.itos), USE_CUDA, n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every


# Begin!
ecs = []
dcs = []
eca = 0
dca = 0

while epoch < n_epochs:
    epoch += 1

    # Get training data for this cycle
    j=0
    for input_batches, input_lengths, target_batches, target_lengths in trn_dataloader:
        print(j)
        #my dirty quick fix, last batch usualyy not full size this avoids error
        if input_batches.size()[1]!=batch_size:
            continue

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder, encoder_optimizer, decoder_optimizer,
            criterion, MAX_LENGTH,batch_size, clip, SOS_token, USE_CUDA )

        # Keep track of loss
        print_loss_total += loss
        #     plot_loss_total += loss
        eca += ec
        dca += dc

        #     job.record(epoch, loss)

        if epoch % print_every == 0:
            #         pdb.set_trace()
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = f'{time_since(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100}%) {print_loss_avg}'
            print(print_summary)

        j+=1

    for val_input_batches, val_input_lengths, val_target_batches, val_target_lengths in valid_dataloader:
        if val_input_batches.size()[1]!=batch_size*2:
            continue
        encoder.train(False)
        decoder.train(False)
        encoder_outputs, encoder_hidden = encoder(val_input_batches, val_input_lengths, None)

        # Create starting vectors for decoder
        #decoder_input = torch.LongTensor([SOS_token])  # SOS
        decoder_input = torch.LongTensor([SOS_token]*batch_size*2)  # SOS
        decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH + 1, MAX_LENGTH + 1)

        # Run through decoder
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            #if using batch have to fix it
            decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[ni.item()])  # another change

            # Next input is chosen word
            decoder_input = torch.LongTensor([ni])
            if USE_CUDA: decoder_input = decoder_input.cuda()

        # Set back to training mode
        encoder.train(True)
        decoder.train(True)

      #  if epoch % evaluate_every == 0:
       #     evaluate_randomly(pairs, MAX_LENGTH, input_lang, output_lang, SOS_token, EOS_token, encoder, decoder, USE_CUDA)