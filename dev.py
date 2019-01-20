import collections
import os, torch
import re
import unicodedata
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

import spacy as spacy
from spacy.symbols import ORTH
from torch.utils.data import Dataset, TensorDataset, DataLoader

TOK_XX=['<eos>','<bos>','<unk>']

def partition(a, sz):
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def partition_by_cores(a):
    return partition(a, len(a)//num_cpus() + 1)

def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

class Tokenizer():
    def __init__(self, lang='en'):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.tok = spacy.load(lang)
        for w in ('<eos>','<bos>','<unk>'):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def sub_br(self,x):
        return self.re_br.sub("\n", x)

    #def spacy_tok(self,x):
     #   return [t.text for t in self.tok.tokenizer(self.sub_br(x))]

    def spacy_tok(self,x):
        #spacy tokenizer
        #return [t.text for t in self.tok.tokenizer(self.sub_br(x))]
        #simple split
        return [t for t in (self.sub_br(x).split())]

    re_rep = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        TOK_UP,TOK_SENT,TOK_MIX = ' t_up ',' t_st ',' t_mx '
        res = []
        prev='.'
        re_word = re.compile('\w')
        re_nonsp = re.compile('\S')
        for s in re.findall(r'\w+|\W+', ss):
            res += ([TOK_UP,s.lower()] if (s.isupper() and (len(s)>2))
    #                 else [TOK_SENT,s.lower()] if (s.istitle() and re_word.search(prev))
                    else [s.lower()])
    #         if re_nonsp.search(s): prev = s
        return ''.join(res)

    def proc_text(self, s):
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(ss, lang):
        tok = Tokenizer(lang)
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss, lang='en', ncpus = None):
        ncpus = ncpus or num_cpus()//2
        if ncpus==0:
            ncpus=1
        #with ProcessPoolExecutor(ncpus) as e:
        with ThreadPoolExecutor(ncpus) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang]*len(ss)), [])


sentences=['mis see on', 'kes see veel on']

toks=Tokenizer.proc_all_mp(partition_by_cores(sentences))
print(toks)

## vocab
EOS='<eos>'
BOS='<bos>'
UNK='<unk>'
class Vocab():
    "Contain the correspondance between numbers and tokens and numericalize."
    def __init__(self, itos):
        self.itos = itos
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

    def numericalize(self, t):
        "Convert a list of tokens `t` to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums, sep=' '):
        "Convert a list of `nums` to their tokens."
        return sep.join([self.itos[i] for i in nums])

    def __getstate__(self):
        return {'itos':self.itos}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

    @classmethod
    def create(cls, tokens, max_vocab:int, min_freq:int, TOK_XX=[EOS, BOS, UNK]):
        "Create a vocabulary from a set of tokens."
        freq = collections.Counter(p for o in tokens for p in o)
        itos = [o for o,c in freq.most_common(max_vocab) if c >= min_freq]
        for o in reversed(TOK_XX):
            if o in itos: itos.remove(o)
            itos.insert(0, o)
        return cls(itos)

print(f'toks: {toks}')
voc=Vocab.create(toks,100, 1)
print(f'itos {voc.itos}')
print(f'stoi {voc.stoi}')
print(f' numericalize {voc.numericalize(toks[0])}')
print('tere')

id_list=[voc.numericalize(text) for text in toks]
print(id_list)


def add_special_strings(texts, EOS, BOS):
    if not isinstance(texts, pd.DataFrame):
        texts=list(texts)
        texts = pd.DataFrame(texts)
    texts=f'{BOS} '+texts[0]+f' {EOS}'
    return texts.values

class SeqData():
    def __init__(self, texts, toks, vocab, toks_id, max_vocab=60000,  min_freq=1, TOK_XX=[EOS, BOS, UNK],
                 tokenizer=Tokenizer):
        self.texts=texts
        self.max_vocab=max_vocab
        self.min_freq=min_freq
        self.TOK_XX=TOK_XX
        self.tokenizer=tokenizer
        self.toks=toks
        self.vocab=vocab
        self.toks_id=toks_id

    @classmethod
    def create(cls, texts, max_vocab=60000,  min_freq=1,TOK_XX=[EOS, BOS, UNK], tokenizer=Tokenizer):
        texts=add_special_strings(texts, EOS, BOS)
        toks=tokenizer.proc_all_mp([texts])
        vocab=Vocab.create(toks, max_vocab, min_freq, TOK_XX)
        toks_id=np.array([vocab.numericalize(text) for text in toks])
        return cls(texts, toks, vocab, toks_id, max_vocab, min_freq, TOK_XX, tokenizer)

sentences=['mis see on', 'kes see veel on']
pr=SeqData.create(sentences)
print(pr.toks)

def to_padded_tensor(toks_id, pad_end=True, pad_idx=1):
    lens=[len(t) for t in toks_id]
    max_len=max(lens)
    res = torch.zeros(len(toks_id), max_len).long() + pad_idx
    for i, toks in enumerate(toks_id):
        if pad_end:
            res[i,0:len(toks)] = torch.LongTensor(toks)
        else:
            res[i, -len(toks):] = torch.LongTensor(toks)
    return res, lens


def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

class Seq2SeqDataset(Dataset):
    """helper to wrap datasets for dataloader"""
    def __init__(self, x, y):
        self.x,self.y = x,y
    def __getitem__(self, idx):
        return A(self.x[idx], self.y[idx])
    def __len__(self):
        return len(self.x)

def to_padded_tensor(sequences, pad_end=True, pad_idx=1, transpose=True):
    lens=[len(seq) for seq in sequences]
    max_len=max(lens)
    tens = torch.zeros(len(sequences), max_len).long() + pad_idx
    for i, toks in enumerate(sequences):
        if pad_end:
            tens[i, 0:len(toks)] = torch.LongTensor(toks)
        else:
            tens[i, -len(toks):] = torch.LongTensor(toks)
    if transpose:
        tens=tens.transpose(0, 1)
    return tens, lens


def collate_fn(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    #data=np.array(data)
    input_seq, target_seq=zip(*data)
    input_tens, input_lens=to_padded_tensor(input_seq)
    target_tens, target_lens =to_padded_tensor(target_seq)
    return input_tens, input_lens, target_tens, target_lens

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def read_pairs_txt(filename):
    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    return pairs

class Seq2SeqDataManager():
   # def __init__(self, texts_x, texts_y, seq_x, seq_y,seq_x_tensor, lens_x, seq_y_tensor, lens_y, min_freq=1,
    #             max_vocab=60000, TOK_XX=[EOS, BOS, UNK], tokenizer=Tokenizer, valid_perc=0.1):
    def __init__(self, texts_x, texts_y, seq_x, seq_y, min_freq=1,
                     max_vocab=60000, TOK_XX=[EOS, BOS, UNK], tokenizer=Tokenizer, valid_perc=0.1):
        self.texts_x=texts_x
        self.texts_y=texts_y
        self.seq_x=seq_x
        self.seq_y=seq_y
       # self.seq_x_tensor=seq_x_tensor
        #self.lens_x=lens_x
        #self.seq_y_tensor=seq_y_tensor
        #self.lens_y=lens_y
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.TOK_XX = TOK_XX
        self.tokenizer = tokenizer
        self.valid_perc=valid_perc

    def get_dataloaders(self, batch_size=10, seed=1):
        self.seed=seed
        self.batch_size=batch_size
        np.random.seed(self.seed)

        idxs=np.random.choice(np.array(range(len(self.seq_x.toks_id))), size=len(self.seq_x.toks_id), replace=False)
        valid_end_idx=int(self.valid_perc*len(idxs))

        self.train_idxs=idxs[valid_end_idx:]
        self.valid_idxs=idxs[0:valid_end_idx]

        train_dataset=Seq2SeqDataset(self.seq_x.toks_id[self.train_idxs], self.seq_y.toks_id[self.train_idxs])
        valid_dataset=Seq2SeqDataset(self.seq_x.toks_id[self.valid_idxs], self.seq_y.toks_id[self.valid_idxs])
        #train_dataset=TensorDataset(self.seq_x_tensor[self.train_idxs], self.seq_y_tensor[self.train_idxs])
        #valid_dataset=TensorDataset(self.seq_x_tensor[self.valid_idxs], self.seq_y_tensor[self.valid_idxs])
        train_dataloader=DataLoader(train_dataset, batch_size=self.batch_size,collate_fn=collate_fn)
        valid_dataloader=DataLoader(valid_dataset, batch_size=self.batch_size*2, collate_fn=collate_fn)
        return train_dataloader, valid_dataloader

    @classmethod
    def create(cls, texts_x, texts_y, min_freq=1, max_vocab=60000, TOK_XX=[EOS, BOS, UNK], tokenizer=Tokenizer, valid_perc=0.1):
        seq_x = SeqData.create(texts_x, max_vocab, min_freq, TOK_XX, tokenizer)
        seq_y = SeqData.create(texts_y, max_vocab, min_freq, TOK_XX, tokenizer)
        if len(seq_y.toks_id)!=len(seq_x.toks_id):
            print('source and target sequences have different lengths')
            return
        #seq_x_tensor, lens_x=to_padded_tensor(seq_x.toks_id)
        #seq_y_tensor, lens_y=to_padded_tensor(seq_y.toks_id)
        return cls(texts_x, texts_y, seq_x, seq_y, min_freq, max_vocab,
                   TOK_XX, tokenizer, valid_perc)
        #return cls(texts_x, texts_y, seq_x, seq_y,seq_x_tensor, lens_x, seq_y_tensor, lens_y, min_freq, max_vocab,
         #          TOK_XX, tokenizer, valid_perc)
    @classmethod
    def create_from_txt(cls, filename, min_freq=1, max_vocab=60000, TOK_XX=[EOS, BOS, UNK], tokenizer=Tokenizer, valid_perc=0.1):
        pairs=read_pairs_txt(filename)
        texts_x, texts_y=zip(*pairs)
        seq_x = SeqData.create(texts_x, max_vocab, min_freq, TOK_XX, tokenizer)
        seq_y = SeqData.create(texts_y, max_vocab, min_freq, TOK_XX, tokenizer)
        if len(seq_y.toks_id) != len(seq_x.toks_id):
            print('source and target sequences have different lengths')
            return
        return cls(texts_x, texts_y, seq_x, seq_y, min_freq, max_vocab,
                   TOK_XX, tokenizer, valid_perc)


input_sentences=['mis see on', 'kes see veel on', 'miks on', 'kas on', 'kas on', 'kes on ','mis see on', 'kes see veel on', 'miks on', 'kas on', 'kas on', 'kes on ']
target_sentences=['mis', 'kes', 'miks', 'kas', 'kas', 'kes ','mis', 'kes see veel', 'miks', 'kas', 'kas', 'kes']
pr=Seq2SeqDataManager.create(input_sentences, target_sentences)
trn_dataloader, valid_dataloader=pr.get_dataloaders()
for inp_tens, input_lens, targ_tens, targ_lens in trn_dataloader:
    print(input_lens)
    print(targ_tens)
print(pr)
#def pad_

