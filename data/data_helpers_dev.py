from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from dev import Tokenizer, partition_by_cores

SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeqData:
    def __init__(self, name=''):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = len(self.index2word)  # Count SOS and EOS
        self.ids = np.array([])

    def add_sentence_list(self, sentence_lst):
        self.ids = np.array([self.add_sentence(sent, True) for sent in sentence_lst])

    def add_sentence(self, sentence, return_id=False):
        lst = [self.add_word(word, return_id) for word in sentence.split(' ')]
        if return_id:
            return lst

    def add_word(self, word, return_id=False):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1
        if return_id:
            return self.word2index[word]


class Seq2SeqData(Dataset):
    """class for x and y to be processed same time"""

    def __init__(self, x, y, val_per=.1, random_seed=1):
        #         np.random.seed(random_seed)
        self.val_per = val_per
        self.x_seq = SeqData()
        self.y_seq = SeqData()
        self.x_seq.add_sentence_list(x)
        self.y_seq.add_sentence_list(y)

        #         self.val_ids=np.random.randint(low=0, high=len(self.x_seq.ids), size=len(self.x_seq.ids)*self.val_per)

        self.x = self.x_seq.ids
        self.y = self.y_seq.ids

    def __getitem__(self, idx):
        # add eos item to the end, might need more elaborate method
        # dtype=torch.long, device=device
        return (torch.tensor(self.x[idx] + [EOS_token], dtype=torch.long, device=device).view(-1, 1),
                torch.tensor(self.y[idx] + [EOS_token], dtype=torch.long, device=device).view(-1, 1))

    def __len__(self):
        return len(self.x)



