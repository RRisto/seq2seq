import time

import torch
import torch.nn as nn
from torch import optim
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from seq2seq.data.data_manager import to_padded_tensor
from seq2seq.data.tokenizer import TOK_XX, normalize_string
from seq2seq.model.decoder import LuongAttnDecoderRNN
from seq2seq.model.encoder import EncoderRNN
from seq2seq.utils.masked_cross_entropy import masked_cross_entropy


class Seq2seqLearner(nn.Module):
    def __init__(self, data_manager, hidden_size, n_layers=2, dropout= 0.1, emb_vecs_x=None, emb_vecs_y=None, attn_model= 'dot'):
        super(Seq2seqLearner, self).__init__()
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.dropout=dropout
        self.data_manager=data_manager
        #self.emb_vecs_x=emb_vecs_x
        #self.emb_vecs_y=emb_vecs_y
        self.attn_model=attn_model

        self.encoder=EncoderRNN(self.data_manager.itos_x, self.hidden_size, self.n_layers, self.dropout, emb_vecs_x)
        self.decoder= LuongAttnDecoderRNN(self.attn_model, self.data_manager.itos_y, self.hidden_size, self.n_layers, self.dropout, emb_vecs_y)


    def forward(self, input_batches, input_lengths, target_batches, target_lengths, device='cpu', return_attention=False):

        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)
        # Prepare input and output variables
        batch_size=input_batches.size(1)
        decoder_input = torch.tensor([TOK_XX.BOS_id] * batch_size, device=device)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # Use last (forward) hidden state from encoder

        max_target_length = max(target_lengths)
        all_decoder_outputs = torch.zeros(max_target_length, batch_size, self.decoder.output_size, device=device)

        if return_attention:
            # Store output words and attention states
            decoded_words = [[] for _ in range(batch_size)]
            decoder_attentions = torch.zeros(batch_size, self.max_len + 1, self.max_len + 1)

        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t]  # Next in

            if return_attention:
                decoder_attentions[:, t, :decoder_attention.size(2)] += decoder_attention.squeeze(1).cpu().data
                # Choose top word from output
                topv, topi = decoder_output.data.topk(1)
                for i in range(len(topv)):
                    ni = topi[i][0]
                    if ni.item() == TOK_XX.EOS_id:
                        decoded_words[i].append(TOK_XX.EOS)
                    else:
                        #todo make datamanager wrapper for this call
                        decoded_words[i].append(self.data_manager.train_seq2seq.seq_y.vocab.itos[ni.item()])
                # Next input is chosen word
                decoder_input = topi.squeeze().clone().detach()

        if return_attention:
            all_decoder_outputs=all_decoder_outputs, decoder_attentions[0, :t + 1, :len(encoder_outputs)], decoded_words

        return all_decoder_outputs

    def fit(self, n_epochs, learning_rate = 0.001, decoder_learning_ratio = 5.0, train_batch_size=100,
            valid_batch_size=100, clip = 50.0, teacher_forcing_ratio = 0.5, show_attention_every=5, device='cpu'):
        self.learning_rate=learning_rate
        self.decoder_learning_ratio=decoder_learning_ratio
        self.train_batch_size=train_batch_size
        self.valid_batch_size=valid_batch_size
        self.clip=clip
        self.teacher_forcing_ratio=teacher_forcing_ratio
        self.max_len=self.data_manager.max_ntoks

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate * self.decoder_learning_ratio)
        self.loss_func = masked_cross_entropy

        train_dataloader, valid_dataloader = self.data_manager.get_dataloaders(train_batch_size=self.train_batch_size,
                                                                          valid_batch_size=self.valid_batch_size)

        eca = 0
        dca = 0

        for epoch in range(n_epochs):
            self.encoder.train(True)
            self.decoder.train(True)

            start = time.time()
            loss_total_train = 0
            nbatches_train = 0
            #train_batch_size = self.batch_size
            for input_batches, input_lengths, target_batches, target_lengths in train_dataloader:
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                all_decoder_outputs = self.forward(
                    input_batches, input_lengths, target_batches, target_lengths, device)

                loss = self.loss_func(
                    all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
                    target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
                    target_lengths)
                loss.backward()

                # Clip gradient norms
                ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
                dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
                # Update parameters with optimizers
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                nbatches_train += 1
                loss_total_train += loss
                eca += ec
                dca += dc
            train_loss_avg = loss_total_train / nbatches_train

            self.encoder.train(False)
            self.decoder.train(False)
            loss_total_valid = 0
            nbatches_valid = 0

            for val_input_batches, val_input_lengths, val_target_batches, val_target_lengths in valid_dataloader:
                val_all_decoder_outputs, decoder_attentions, decoded_words = self.forward(val_input_batches,
                                                                                          val_input_lengths,
                                                                                          val_target_batches,
                                                                                          val_target_lengths, device, True)
                with torch.no_grad():
                    loss_valid = self.loss_func(
                        val_all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
                        val_target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
                        val_target_lengths)
                nbatches_valid += 1
                loss_total_valid += loss_valid

            valid_loss_avg = loss_total_valid / nbatches_valid
            self.print_summary(start, epoch, n_epochs, train_loss_avg.item(), valid_loss_avg.item())

            if epoch % show_attention_every == 0:
                self.show_attention(self.data_manager.valid_seq2seq.seq_x.vocab.textify(val_input_batches.t()[0]),
                               decoded_words[0], decoder_attentions)

    def time_since(self, start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        return hours, minutes, seconds

    def print_summary(self, start, epoch, epochs, loss_train, loss_valid):
        end = time.time()
        hours, minutes, seconds = self.time_since(start, end)
        summary = f'{round(hours)}:{round(minutes)}:{round(seconds, 2)} ({epoch + 1} ' \
            f'{round((epoch + 1) / epochs * 100, 2)}%) ' \
            f'loss train: {round(loss_train, 3)} loss valid: {round(loss_valid, 3)}'
        print(summary)


    def show_attention(self, input_sentence, output_words, attentions):
        df_attentions = pd.DataFrame(attentions.numpy())
        df_attentions.index = output_words
        df_attentions.columns = input_sentence.split(' ')

        ax = sns.heatmap(df_attentions, cmap='Blues', robust=True, cbar=False, cbar_kws={"orientation": "horizontal"})
        ax.xaxis.tick_top()  # x axis on top
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('input')
        ax.set_ylabel('output')
        plt.show()
        plt.close()


    def predict(self, text, device='cpu'):
        self.encoder.train(False)
        self.decoder.train(False)
        text = normalize_string(text)
        input_toks_id = self.data_manager.train_seq2seq.seq_x.numericalize(text)
        input_batch, input_length = to_padded_tensor([input_toks_id], device=device)

        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_length, None)
        decoder_input = torch.tensor([TOK_XX.BOS_id], device=device)  # SOS
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        decoded_toks_id = []
        for di in range(self.max_len):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
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
        text = self.data_manager.train_seq2seq.seq_y.textify(decoded_toks_id)
        return text



