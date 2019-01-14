import os
import re
from concurrent.futures.process import ProcessPoolExecutor

import spacy as spacy
from spacy.symbols import ORTH


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
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang]*len(ss)), [])


sentences=['mis see on ', 'kes see veel on']

toks=Tokenizer.proc_all_mp(partition_by_cores(sentences))
print(toks)