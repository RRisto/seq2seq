import time
import torch.nn as nn
from torch import optim

from masked_cross_entropy import *
from data_helpers import prepare_data, random_batch
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
input_lang, output_lang, pairs = prepare_data('eng', 'fra',MIN_LENGTH, MAX_LENGTH, True)

input_lang.trim(MIN_COUNT)
output_lang.trim(MIN_COUNT)

keep_pairs = []

#filter pairs
for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True

    for word in input_sentence.split(' '):
        if word not in input_lang.word2index:
            keep_input = False
            break

    for word in output_sentence.split(' '):
        if word not in output_lang.word2index:
            keep_output = False
            break

    # Remove if pair doesn't match input and output conditions
    if keep_input and keep_output:
        keep_pairs.append(pair)

print(f"Trimmed from {len(pairs)} pairs to {len(keep_pairs)}, {round((len(keep_pairs) / len(pairs))*100,2)}% of total")
pairs = keep_pairs

##test
small_batch_size = 3
input_batches, input_lengths, target_batches, target_lengths = random_batch(small_batch_size, pairs, input_lang,
                                                                            output_lang, PAD_token, EOS_token, USE_CUDA)

print('input_batches', input_batches.size()) # (max_len x batch_size)
print('target_batches', target_batches.size()) # (max_len x batch_size)

small_hidden_size = 8
small_n_layers = 2

encoder_test = EncoderRNN(input_lang.n_words, small_hidden_size, small_n_layers)
decoder_test = LuongAttnDecoderRNN('general', small_hidden_size, output_lang.n_words, USE_CUDA, small_n_layers)

encoder_outputs, encoder_hidden = encoder_test(input_batches, input_lengths, None)

print('encoder_outputs', encoder_outputs.size()) # max_len x batch_size x hidden_size
print('encoder_hidden', encoder_hidden.size()) # n_layers * 2 x batch_size x hidden_size

max_target_length = max(target_lengths)

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
    decoder_input = target_batches[t] # Next input is current target

# Test masked cross entropy loss
loss = masked_cross_entropy(
    all_decoder_outputs.transpose(0, 1).contiguous(),
    target_batches.transpose(0, 1).contiguous(),
    target_lengths,
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
batch_size = 3

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
# n_epochs = 50000
n_epochs = 1000
epoch = 0
# plot_every = 20
plot_every = 2000
# print_every = 100
print_every = 500
# evaluate_every = 1000
evaluate_every = 300

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, USE_CUDA, n_layers, dropout=dropout)

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
    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size,pairs, input_lang,output_lang,
                                                                                PAD_token, EOS_token, USE_CUDA)

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

    if epoch % evaluate_every == 0:
        evaluate_randomly(pairs, MAX_LENGTH, input_lang, output_lang, SOS_token, EOS_token, encoder, decoder, USE_CUDA)