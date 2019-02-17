from torch import optim
import torch.nn as nn

from seq2seq.utils.masked_cross_entropy import masked_cross_entropy
from seq2seq.data.data_manager import Seq2SeqDataManager
from seq2seq.model.decoder import LuongAttnDecoderRNN
from seq2seq.model.encoder import EncoderRNN
from seq2seq.train import fit, predict

DEVICE = 'cpu'
MIN_LENGTH = 3
# MAX_LENGTH = 25
MAX_LENGTH = 10
MIN_COUNT = 3

## Get data
# data_manager=Seq2SeqDataManager.create_from_txt('data/eng-fra_sub.txt')
data_manager = Seq2SeqDataManager.create_from_txt('data/eng-fra_sub.txt', min_freq=MIN_COUNT, min_ntoks=MIN_LENGTH,
                                                  max_ntoks=MAX_LENGTH, switch_pair=True, device=DEVICE)
# data_manager=Seq2SeqDataManager.create_from_txt('data/eng-fra.txt', min_ntoks=3, max_ntoks=10)
##test
train_batch_size = 100
valid_batch_size = 100
train_dataloader, valid_dataloader = data_manager.get_dataloaders(train_batch_size=train_batch_size,
                                                                  valid_batch_size=valid_batch_size)

##model conf
attn_model = 'dot'
# hidden_size = 500
hidden_size = 50
n_layers = 2
dropout = 0.1

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
# learning_rate = 0.0001
learning_rate = 0.001
decoder_learning_ratio = 5.0
# n_epochs = 50000
n_epochs = 20
epoch = 0
# plot_every = 20
plot_every = 2000
# print_every = 100

# evaluate_every = 1000
evaluate_every = 1

# Initialize models
encoder = EncoderRNN(len(data_manager.train_seq2seq.seq_x.vocab.itos), hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, hidden_size, len(data_manager.train_seq2seq.seq_y.vocab.itos), n_layers,
                              dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# test
fit(n_epochs, encoder, encoder_optimizer, decoder, decoder_optimizer, data_manager, train_batch_size, valid_batch_size,
    clip, MAX_LENGTH + 1,
    masked_cross_entropy, train_dl=train_dataloader, valid_dl=valid_dataloader, device=DEVICE)

original_xtext = 'Je suis sûr.'
original_ytext = 'I am sure.'
predicted_text = predict(original_xtext, encoder, decoder, data_manager, device=DEVICE)
print(f'original text: {original_xtext}')
print(f'original answer: {original_ytext}')
print(f'predicted text: {predicted_text}')
