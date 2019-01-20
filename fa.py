from fastai.text import *
from fastai import *

path = untar_data(URLs.IMDB_SAMPLE)
path

df = pd.read_csv(path/'texts.csv')
df.head()


# Language model data
data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
# Classifier model data
data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)

data_lm.save()
data_clas.save()