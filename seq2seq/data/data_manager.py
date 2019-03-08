import collections, torch
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from seq2seq.data.tokenizer import Tokenizer, TOK_XX
from seq2seq.data.utils import normalize_string, load_ft_vectors


class Vocab():
    """Contain the correspondence between numbers and tokens and numericalize.
    Is from fastai: https://github.com/fastai/fastai/blob/master/fastai/text/transform.py#L121"""

    def __init__(self, itos:list):
        self.itos = itos
        self.stoi = collections.defaultdict(int, {v: k for k, v in enumerate(self.itos)})

    def numericalize(self, tokens:list):
        "Convert a list of tokens  to their ids."
        return [self.stoi[w] for w in tokens]

    def textify(self, nums:np.ndarray, sep:str=' '):
        "Convert a list of `nums` to their tokens."
        return sep.join([self.itos[i] for i in nums])

    def keep_tokens(self, toks_keep:list):
        self.itos = [tok for tok in self.itos if tok in toks_keep]
        ids_to_remove = [id for id in self.stoi.keys() if id not in self.itos]
        for id in ids_to_remove:
            del self.stoi[id]

    @classmethod
    def create(cls, tokens:list, max_vocab: int, min_freq: int, TOK_XX:TOK_XX=TOK_XX):
        "Create a vocabulary from a set of tokens."
        freq = collections.Counter(p for o in tokens for p in o)
        itos = [o for o, c in freq.most_common(max_vocab) if c >= min_freq]
        for o in reversed(TOK_XX.TOK_XX):
            if o in itos:
                itos.remove(o)
            itos.insert(0, o)
        return cls(itos)


def add_special_strings(texts:list, BOS:str=None, EOS:str=None):
    """for list/df of texts add string to the beginning (BOS) or to the end (EOS)"""
    if not isinstance(texts, pd.Series):
        texts = list(texts)
        texts = pd.Series(texts)
    if BOS is not None:
        texts = f'{BOS} ' + texts
    if EOS is not None:
        texts = texts + f' {EOS}'
    return texts.values


class SeqData():
    """class for texts sequences, tokenizes and numericalizes them"""

    def __init__(self, texts:np.ndarray, toks:list, vocab:Vocab, toks_id:np.ndarray, max_vocab:int=60000,
                 min_freq:int=1, TOK_XX:TOK_XX=TOK_XX, tokenizer:Tokenizer=Tokenizer(lang='en')):
        self.texts = texts
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.TOK_XX = TOK_XX
        self.tokenizer = tokenizer
        self.toks = toks
        self.vocab = vocab
        self.toks_id = toks_id

    def numericalize(self, text:str):
        text = add_special_strings([text], self.TOK_XX.BOS, self.TOK_XX.EOS)
        toks = self.tokenizer.proc_all_mp([text[0]])
        toks = self.vocab.numericalize(toks[0])
        return toks

    def textify(self, tok_ids:np.ndarray, sep:str=' '):
        text = self.vocab.textify(tok_ids, sep=sep)
        return text

    def toks_lens(self):
        return np.array([len(toks) for toks in self.toks_id])

    def toks_contain_unk(self):
        return np.array([True if self.TOK_XX.UNK_id in toks else False for toks in self.toks_id])

    def ids_to_keep(self, max_len:int, min_len:int=1, remove_unk:bool=True):
        self.min_len = min_len
        self.max_len = max_len
        toks_len = self.toks_lens()
        if remove_unk:
            toks_cont_unk = self.toks_contain_unk()
            idx_to_keep = (toks_len >= self.min_len) & (toks_len <= self.max_len) & (~toks_cont_unk)
        else:
            idx_to_keep = (toks_len >= self.min_len) & (toks_len <= self.max_len)
        return idx_to_keep

    def remove(self, max_len:int=None, min_len:int=None, idxs_to_keep:np.ndarray=None, remove_unk:bool=True,
               valid:bool=False, train_vocab:Vocab=None):
        """remove by length or indexes (boolean), additional option remove sequences that have unknown tokens
        also removes tokens that no longer in corpus from dictionary. if is validation sequence,
        sets it vocab state ase train vocab. Removing unknown might leave some unknwon because some words
        might have lower freq after removing some sequences"""
        if max_len is not None and min_len is not None and idxs_to_keep is None:
            idxs_to_keep = self.ids_to_keep(max_len, min_len, remove_unk)
        if idxs_to_keep is not None:
            n_seq_original = len(self.toks_id)
            self.texts = self.texts[idxs_to_keep]
            self.toks_id = self.toks_id[idxs_to_keep]
            self.toks = np.array(self.toks)[idxs_to_keep].tolist()
            if valid and train_vocab is not None:
                self.vocab = train_vocab

            print(f'    kept {sum(idxs_to_keep)} sequences from {n_seq_original} sequences')
        else:
            print('You must have max_len and min_len values set or idx_to_keep set')

    @classmethod
    def create(cls, texts:np.ndarray, tokenizer:Tokenizer, max_vocab:int=60000, min_freq:int=1, TOK_XX:TOK_XX=TOK_XX,
               vocab:Vocab=None, add_EOS:bool=True):
        if add_EOS:
            texts = add_special_strings(texts, None, TOK_XX.EOS)
        toks = tokenizer.proc_all_mp(texts.tolist())
        if vocab is None:
            vocab = Vocab.create(toks, max_vocab, min_freq, TOK_XX)
        toks_id = np.array([vocab.numericalize(text) for text in toks])
        return cls(texts, toks, vocab, toks_id, max_vocab, min_freq, TOK_XX, tokenizer)


def A(*a:tuple):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a) == 1 else [np.array(o) for o in a]


class Seq2SeqDataset(Dataset):
    """helper to wrap x and y sequences for torch dataloader,
    keeps correspondence that samples that are remove from x are also removed from y"""

    def __init__(self, seq_x:SeqData, seq_y:SeqData, min_ntoks:int, max_ntoks:int):
        self.seq_x = seq_x
        self.seq_y = seq_y
        self.x = self.seq_x.toks_id
        self.y = self.seq_y.toks_id
        self.min_ntoks = min_ntoks
        self.max_ntoks = max_ntoks

    def __getitem__(self, idx):
        return A(self.x[idx], self.y[idx])

    def __len__(self):
        return len(self.x)

    def textify(self, tok_ids,y=True,  sep:str=' '):
        seq=self.seq_y if y else self.seq_x
        return seq.textify(tok_ids, sep=sep)

    @classmethod
    def create(cls, seq_x:SeqData, seq_y:SeqData, min_ntoks:int, max_ntoks:int, remove_unk:bool=False, valid:bool=False,
               train_seq2seq_xvocab:Vocab=None, train_seq2seq_yvocab:Vocab=None):
        print('  Checking x sequence match to max, min criterias')
        idxs_to_keep = seq_x.ids_to_keep(max_ntoks, min_ntoks, remove_unk)
        seq_x.remove(idxs_to_keep=idxs_to_keep, valid=valid, train_vocab=train_seq2seq_xvocab)
        seq_y.remove(idxs_to_keep=idxs_to_keep, valid=valid, train_vocab=train_seq2seq_yvocab)

        print('  Checking y sequence match to max, min criterias')
        idxs_to_keep = seq_y.ids_to_keep(max_ntoks, min_ntoks, remove_unk)
        seq_x.remove(idxs_to_keep=idxs_to_keep, valid=valid, train_vocab=train_seq2seq_xvocab)
        seq_y.remove(idxs_to_keep=idxs_to_keep, valid=valid, train_vocab=train_seq2seq_yvocab)

        seq_x = SeqData.create(seq_x.texts, seq_x.tokenizer, seq_x.max_vocab, seq_x.min_freq, seq_x.TOK_XX,
                               train_seq2seq_xvocab, add_EOS=False)
        seq_y = SeqData.create(seq_y.texts, seq_y.tokenizer, seq_y.max_vocab, seq_y.min_freq, seq_y.TOK_XX,
                               train_seq2seq_yvocab, add_EOS=False)
        return cls(seq_x, seq_y, min_ntoks, max_ntoks)


class Seq2SeqDataManager():
    """class to manage x and y strain and validation sequences creation. Helps to create dataloaders"""

    def __init__(self, train_seq2seq:Seq2SeqDataset, valid_seq2seq:Seq2SeqDataset, lang_x:str, lang_y:str, device:str, max_ntoks:int):
        self.train_seq2seq = train_seq2seq
        self.valid_seq2seq = valid_seq2seq
        self.lang_x=lang_x
        self.lang_y=lang_y
        self.device = device
        self.max_ntoks=max_ntoks
        self.itos_x=self.train_seq2seq.seq_x.vocab.itos
        self.itos_y=self.train_seq2seq.seq_y.vocab.itos
        self.vectors_x={}
        self.vectors_y={}

    @staticmethod
    def _to_padded_tensor(sequences:tuple, pad_end:bool=True, pad_idx:int=TOK_XX.PAD_id, transpose:bool=True,
                          device:str='cpu'):
        """turns sequences of token ids into tensor with padding and optionally transpose"""
        lens = torch.tensor([len(seq) for seq in sequences], device=device)
        max_len = max(lens)
        tens = torch.zeros(len(sequences), max_len, device=device).long() + pad_idx
        for i, toks in enumerate(sequences):
            if pad_end:
                tens[i, 0:len(toks)] = torch.tensor(toks)
            else:
                tens[i, -len(toks):] = torch.tensor(toks)
        if transpose:
            tens = tens.transpose(0, 1)
        return tens, lens

    def _collate_fn(self, data:list):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        input_seq, target_seq = zip(*data)
        input_tens, input_lens = self._to_padded_tensor(input_seq, device=self.device)
        target_tens, target_lens = self._to_padded_tensor(target_seq, device=self.device)
        return input_tens, input_lens, target_tens, target_lens

    def get_dataloaders(self, train_batch_size:int=10, valid_batch_size:int=1, device:str='cpu'):
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.device = device

        train_dataloader = DataLoader(self.train_seq2seq, batch_size=self.train_batch_size, collate_fn=self._collate_fn)
        valid_dataloader = DataLoader(self.valid_seq2seq, batch_size=self.valid_batch_size, collate_fn=self._collate_fn)
        return train_dataloader, valid_dataloader

    def textify(self, tok_ids:list, train:bool=True, y:bool=True, sep:str=' '):
        seq=self.train_seq2seq if train else self.valid_seq2seq
        return seq.textify(tok_ids, y, sep)

    def load_ft_vectors(self, filename_x:str=None, filename_y:str=None):
        if filename_x is not None:
            self.vectors_x=load_ft_vectors(filename_x, self.itos_x)
        if filename_y is not None:
            self.vectors_y=load_ft_vectors(filename_y, self.itos_y)

    def save(self, filename:str):
        #this is needed because spacy tokenizer has some issues with pickling
        self.train_seq2seq.seq_x.tokenizer=None
        self.train_seq2seq.seq_y.tokenizer=None
        self.valid_seq2seq.seq_x.tokenizer=None
        self.valid_seq2seq.seq_y.tokenizer=None
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        self.train_seq2seq.seq_x.tokenizer = Tokenizer(self.lang_x)
        self.train_seq2seq.seq_y.tokenizer = Tokenizer(self.lang_y)
        self.valid_seq2seq.seq_x.tokenizer = Tokenizer(self.lang_x)
        self.valid_seq2seq.seq_y.tokenizer = Tokenizer(self.lang_y)

    @staticmethod
    def load(filename:str):
        with open(filename, 'rb') as f:
            data_manager=pickle.load(f)
            data_manager.train_seq2seq.seq_x.tokenizer = Tokenizer(data_manager.lang_x)
            data_manager.train_seq2seq.seq_y.tokenizer = Tokenizer(data_manager.lang_y)
            data_manager.valid_seq2seq.seq_x.tokenizer = Tokenizer(data_manager.lang_x)
            data_manager.valid_seq2seq.seq_y.tokenizer = Tokenizer(data_manager.lang_y)
            return data_manager

    @classmethod
    def create(cls, train_x: pd.Series, train_y:pd.Series, valid_x:pd.Series, valid_y:pd.Series, lang_x:str, lang_y:str,
               min_freq:int=2, max_vocab:int=60000, min_ntoks:int=1, max_ntoks:int=7, TOK_XX:TOK_XX=TOK_XX,
               device:str='cpu'):
        tokenizer_x=Tokenizer(lang_x)
        tokenizer_y=Tokenizer(lang_y)
        train_seq_x = SeqData.create(train_x, tokenizer_x, max_vocab, min_freq, TOK_XX)
        train_seq_y = SeqData.create(train_y, tokenizer_y, max_vocab, min_freq, TOK_XX)

        if len(train_seq_y.toks_id) != len(train_seq_x.toks_id):
            raise ValueError('source and target sequences have different number of documents/texts')

        valid_seq_x = SeqData.create(valid_x, tokenizer_x, max_vocab, min_freq, TOK_XX, train_seq_x.vocab)
        valid_seq_y = SeqData.create(valid_y, tokenizer_y, max_vocab, min_freq, TOK_XX, train_seq_y.vocab)

        print('Creating training dataset')
        train_seq2seq = Seq2SeqDataset.create(train_seq_x, train_seq_y, min_ntoks, max_ntoks)
        print('Creating valid dataset')
        valid_seq2seq = Seq2SeqDataset.create(valid_seq_x, valid_seq_y, min_ntoks, max_ntoks, False, True,
                                              train_seq2seq.seq_x.vocab, train_seq2seq.seq_y.vocab)
        return cls(train_seq2seq, valid_seq2seq, lang_x, lang_y, device, max_ntoks)

    @classmethod
    def create_from_txt(cls, train_filename:str, lang_x:str, lang_y:str, valid_filename:str=None, min_freq:int=2,
                        max_vocab:int=60000, min_ntoks:int=1, max_ntoks:int=7, TOK_XX:TOK_XX=TOK_XX,
                        valid_perc:float=.1, seed:int=1, switch_pair:bool=True, device:str='cpu'):
        """if valid has filename loads validation set from there, otherwise makes valid set from train set using
        valid perc"""

        train_df = pd.read_table(train_filename, header=None)
        if switch_pair:
            train_df = train_df[[1, 0]]
            train_df.columns = pd.Index([0, 1])
        train_x = train_df[0].apply(lambda x: normalize_string(x))
        train_y = train_df[1].apply(lambda x: normalize_string(x))
        if valid_filename is not None:
            valid_df = pd.read_table(valid_filename, header=None)
            if switch_pair:
                valid_df = valid_df[[1, 0]]
                valid_df.columns = pd.Index([0, 1])
            valid_x = valid_df[0].apply(lambda x: normalize_string(x))
            valid_y = valid_df[1].apply(lambda x: normalize_string(x))
        else:
            np.random.seed(seed)
            rands = np.random.rand(len(train_x))
            valid_x = train_x[rands <= valid_perc]
            valid_y = train_y[rands <= valid_perc]

            train_x = train_x[rands > valid_perc]
            train_y = train_y[rands > valid_perc]

        return cls.create(train_x, train_y, valid_x, valid_y, lang_x, lang_y, min_freq, max_vocab, min_ntoks, max_ntoks,
                          TOK_XX, device)
