import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

from seq2seq.data.data_manager import to_padded_tensor, normalize_string, TOK_XX
from seq2seq.utils.masked_cross_entropy import *
from seq2seq.utils.utils import print_summary


def train_batch(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
                decoder_optimizer, batch_size, clip, MAX_LENGTH, loss_func=masked_cross_entropy, TOK_XX=TOK_XX,
                device='cpu'):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    # Prepare input and output variables
    decoder_input = torch.tensor([TOK_XX.BOS_id] * batch_size, device=device)
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size, device=device)

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = loss_func(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths)
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data, ec, dc


def valid_batch(encoder, decoder, data_manager, val_input_batches, val_input_lengths, val_target_batches,
                val_target_lengths, valid_batch_size, MAX_LENGTH, TOK_XX=TOK_XX, device='cpu'):
    encoder_outputs, encoder_hidden = encoder(val_input_batches, val_input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = torch.tensor([TOK_XX.BOS_id] * valid_batch_size, device=device)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    # Run through decoder
    val_target_max_len = max(val_target_lengths)
    all_decoder_outputs = torch.zeros(val_target_max_len, valid_batch_size, decoder.output_size, device=device)

    # Store output words and attention states
    decoded_words = [[]] * valid_batch_size
    decoder_attentions = torch.zeros(valid_batch_size, MAX_LENGTH + 1, MAX_LENGTH + 1)

    for di in range(val_target_max_len):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
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
                decoded_words[i].append(data_manager.train_seq2seq.seq_y.vocab.itos[ni.item()])
        # Next input is chosen word
        decoder_input = torch.tensor(topi.squeeze(), device=device)

        with torch.no_grad():
            loss = masked_cross_entropy(
                all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
                val_target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
                val_target_lengths)

    return loss, decoder_attentions[0, :di + 1, :len(encoder_outputs)], decoded_words
    return loss, decoder_attentions, decoded_words



def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    #df_attentions=pd.DataFrame(attentions.numpy())
    #df_attentions.index=output_words
    #df_attentions.columns=input_sentence.split(' ') + ['<EOS>']

    #fig, ax =sns.heatmap(df_attentions, cmap='Blues')

    # Set up axes
    ax.set_xlabel('input')
    ax.set_ylabel('output')

    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    plt.show()
    plt.close()


def fit(epochs, encoder, encoder_optimizer, decoder, decoder_optimizer, data_manager, batch_size=100,
        valid_batch_size=100, clip=50.0,
        MAX_LENGTH=10, loss_func=masked_cross_entropy, train_dl=None, valid_dl=None, TOK_XX=TOK_XX, device='cpu',
        show_attention_every=10):
    eca = 0
    dca = 0

    for epoch in range(epochs):
        encoder.train(True)
        decoder.train(True)

        start = time.time()
        loss_total_train = 0
        nbatches_train = 0
        train_batch_size = batch_size
        for input_batches, input_lengths, target_batches, target_lengths in train_dl:
            if input_batches.size()[1] != batch_size:
                train_batch_size = input_batches.size()[1]

            loss, ec, dc = train_batch(
                input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
                decoder_optimizer, train_batch_size, clip, MAX_LENGTH, loss_func, TOK_XX, device)

            nbatches_train += 1
            loss_total_train += loss
            eca += ec
            dca += dc
        train_loss_avg = loss_total_train / nbatches_train

        encoder.train(False)
        decoder.train(False)
        loss_total_valid = 0
        nbatches_valid = 0

        valid_batch_size_temp = valid_batch_size
        for val_input_batches, val_input_lengths, val_target_batches, val_target_lengths in valid_dl:
            if val_input_batches.size()[1] != valid_batch_size_temp:
                valid_batch_size_temp = val_input_batches.size()[1]

            loss, decoder_attentions, decoded_words = valid_batch(encoder, decoder, data_manager, val_input_batches,
                                                                  val_input_lengths, val_target_batches,
                                                                  val_target_lengths, valid_batch_size_temp,
                                                                  MAX_LENGTH, TOK_XX, device)
            nbatches_valid += 1
            loss_total_valid += loss

        valid_loss_avg = loss_total_valid / nbatches_valid
        print_summary(start, epoch, epochs, train_loss_avg.item(), valid_loss_avg.item())

        if epoch % show_attention_every == 0:
            show_attention(data_manager.valid_seq2seq.seq_x.vocab.textify(val_input_batches[0]), decoded_words[0],
                           decoder_attentions)
            #show_attention(data_manager.valid_seq2seq.seq_x.vocab.textify(val_input_batches[0]), decoded_words[0],
             #              decoder_attentions[0])


def predict(text, encoder, decoder, data_manager, max_length=10, device='cpu'):
    encoder.train(False)
    decoder.train(False)
    text = normalize_string(text)
    input_toks_id = data_manager.train_seq2seq.seq_x.numericalize(text)
    input_batch, input_length = to_padded_tensor([input_toks_id], device=device)

    encoder_outputs, encoder_hidden = encoder(input_batch, input_length, None)
    decoder_input = torch.tensor([TOK_XX.BOS_id], device=device)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    decoded_toks_id = []
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni.item() == TOK_XX.EOS_id:
            decoded_toks_id.append(ni.item())
            break
        else:
            decoded_toks_id.append(ni.item())

        decoder_input = torch.tensor([ni], device=device)
    # turn them into words
    text = data_manager.train_seq2seq.seq_y.textify(decoded_toks_id)
    return text
