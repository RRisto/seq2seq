import io, re, unicodedata


def load_ft_vectors(fname):
    '''from: https://fasttext.cc/docs/en/english-vectors.html'''
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def unicode_to_ascii(string:str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', string)
        if unicodedata.category(c) != 'Mn' )


def normalize_string(string:str):
    """ trim, and remove non-letter characters"""
    string = unicode_to_ascii(string.strip())
    string = re.sub(r"([,.!?])", r" \1 ", string)
    string = re.sub(r"[^a-zA-Z,.!?]+", r" ", string)
    string = re.sub(r"\s+", r" ", string).strip()
    return string
