## Simple batched seq2seq example

This is my version of pytorch example of seq2seq tutorial: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

New features:
     - "more batched" seq2seq, initial example had batch size of 1, here bact size could be determined by user, making training faster
    - validation data. Separate dataset for validation is used, this gives much more accurate picture of model training than just using training data
    - refactored code:
            - using dataloaders
            - separate classes for data managmement
            - learner class for model creation and training
    - data tokenization is multicore, making it faster
    - possibility to use pretrained wordvectors
    
    
See example in example_seq2seq.ipynb


Some of the inspiration and code are from fastai (https://github.com/fastai/fastai), some from the original tutorial ( https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb). I've referred to fastai if some of the code is taken from there).