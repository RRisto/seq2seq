## Updated batched seq2seq example

This is my version of pytorch [seq2seq tutorial](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb).

New features:

- validation data. Separate dataset for validation is used, this gives much more accurate picture of model training than just using training data
- refactored code:
        - using dataloaders
        - separate classes for data management
        - learner class for model creation and training
- data tokenization is multicored
- possibility to use pretrained wordvectors
- added teacher forcing
    
For more details, explanations see my [article](https://medium.com/@RistoHinno/tuned-version-of-seq2seq-tutorial-ddb64db46e2a).
    
See example in example_seq2seq.ipynb. Example data is from IMDB [dataset](https://course.fast.ai/datasets#nlp).


Inspiration and some of the code is from [fastai](https://github.com/fastai/fastai) and from the [original tutorial](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb).