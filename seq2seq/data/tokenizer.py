import os, re, spacy

from spacy.symbols import ORTH
from concurrent.futures import ThreadPoolExecutor


class TOK_XX:
    """class to keep order, values and ds of special tokens. If you want to change values of special tokens id,
    change it order in TOK_XX. Order is same because vocabulary adds these tokens to the beginning"""
    BOS = '<bos>'
    EOS = '<eos>'
    UNK = '<unk>'
    PAD = '<pad>'
    TOK_XX = [UNK, PAD, BOS, EOS]
    TOK_XX_ids = {k: v for v, k in enumerate(TOK_XX)}
    UNK_id = TOK_XX_ids[UNK]
    PAD_id = TOK_XX_ids[PAD]
    BOS_id = TOK_XX_ids[BOS]
    EOS_id = TOK_XX_ids[EOS]


def partition(a, sz):
    """splits iterables a in equal parts of size sz"""
    return [a[i:i + sz] for i in range(0, len(a), sz)]


def partition_by_cores(a, ncpus):
    return partition(a, len(a) // ncpus)


def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


class Tokenizer:
    def __init__(self, lang:str='en'):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.lang = lang
        self.tok = spacy.load(self.lang)
        for w in TOK_XX.TOK_XX:
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def sub_br(self, x:str):
        return self.re_br.sub("\n", x)

    def spacy_tok(self, x:str):
        #this function splits texts
        return [t.text for t in self.tok.tokenizer(self.sub_br(x))]
        # simple split
        #return [t for t in (self.sub_br(x).split())]

    re_rep = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c, cc = m.groups()
        return f' {TK_REP} {len(cc) + 1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = '<tk_wrep>'
        c, cc = m.groups()
        return f' {TK_WREP} {len(cc.split()) + 1} {c} '

    def do_caps(self, ss:str):
        TOK_UP, TOK_SENT, TOK_MIX = ' <t_up> ', ' <t_st> ', ' <t_mx> '
        res = []
        for s in re.findall(r'\w+|\W+', ss):
            res += ([TOK_UP, s.lower()] if (s.isupper() and (len(s) > 2))
                    else [s.lower()])
        return ''.join(res)

    def proc_text(self, string:str):
        string = self.re_rep.sub(self.replace_rep, string)
        string = self.re_word_rep.sub(self.replace_wrep, string)
        string = self.do_caps(string)
        string = re.sub(r'([/#])', r' \1 ', string)
        string = re.sub(' {2,}', ' ', string)
        return self.spacy_tok(string)

    def proc_all(self, strings:list):
        return [self.proc_text(s) for s in strings]

    def proc_all_mp(self, strings:list, ncpus:int=None):
        ncpus = ncpus or num_cpus() // 2
        if ncpus == 0:
            ncpus = 1
        with ThreadPoolExecutor(ncpus) as e:
            return sum(e.map(self.proc_all, partition_by_cores(strings, ncpus)), [])





