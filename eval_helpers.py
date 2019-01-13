import random
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
from data_helpers import indexes_from_sentence

@torch.no_grad()
def evaluate(input_seq, max_length, input_lang, output_lang, SOS_token, EOS_token,encoder, decoder, USE_CUDA):
    #input_lengths = [len(input_seq)] #looks like a bug should be token len
    input_lengths = [len(input_seq.split(' '))]
    input_seqs = [indexes_from_sentence(input_lang, input_seq, EOS_token)]
    input_batches = torch.LongTensor(input_seqs).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    #     pdb.set_trace()
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([SOS_token]) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni.item()])#another change

        # Next input is chosen word
        decoder_input = torch.LongTensor([ni])
        if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

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

    # show_plot_visdom()
    plt.show()
    plt.close()


def evaluate_and_show_attention(input_sentence, max_length, input_lang,output_lang, SOS_token,EOS_token,
                                        encoder, decoder, USE_CUDA, target_sentence=None):
    output_words, attentions = evaluate(input_sentence, max_length, input_lang,output_lang, SOS_token,EOS_token,
                                        encoder, decoder, USE_CUDA)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)

    show_attention(input_sentence, output_words, attentions)

    print(f' input sentence /n {input_sentence}')
    print(f' target sentence /n {target_sentence}')
    print(f' output sentence /n {output_sentence}')


    # Show input, target, output text in visdom
    # win = 'evaluted (%s)' % hostname
    # text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    # vis.text(text, win=win, opts={'title': win})

def evaluate_randomly(pairs, max_length,input_lang, output_lang,SOS_token, EOS_token, encoder,
                                decoder, USE_CUDA):
    [input_sentence, target_sentence] = random.choice(pairs)
    evaluate_and_show_attention(input_sentence, max_length,input_lang, output_lang,SOS_token, EOS_token, encoder,
                                decoder, USE_CUDA, target_sentence)
