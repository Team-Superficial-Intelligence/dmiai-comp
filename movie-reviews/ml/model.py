from collections import defaultdict
import collections
import pickle
import json
from typing import Collection, List
from fastai.data.external import URLs
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from fastai import *
from fastai.text.core import BaseTokenizer, Tokenizer
from fastai.text.data import *
from fastai.callback import *
from transformers import PreTrainedTokenizer, XLNetTokenizer, XLNetConfig


class FastAiTokenizer(BaseTokenizer):
    def __init__(self):
        self.ptt = XLNetTokenizer.from_pretrained('xlnet-large-cased')

    def __call__(self, *args, **kwargs):
        return self

    def save_vocabulary(self):
        self.ptt.save_vocabulary('/workspace/data/')
        self.ptt.save_pretrained('/workspace/data/')

    def tokenizer(self, t: str) -> List[str]:

        CLS = self.ptt.cls_token
        SEP = self.ptt.sep_token
        tokens = self.ptt.tokenize(t)[:self.ptt.max_len - 2]
        tokens = tokens + [SEP] + [CLS]

        return tokens

    def make_vocab(self):
        make_vocab()


# class TransformersVocab(Vocab):
#     def __init__(self, tokenizer: PreTrainedTokenizer):
#         super(TransformersVocab, self).__init__(itos=[])
#         self.tokenizer = tokenizer

#     def numericalize(self, t: Collection[str]) -> List[int]:
#         "Convert a list of tokens `t` to their ids."
#         return self.tokenizer.convert_tokens_to_ids(t)

#     def textify(self, nums: Collection[int], sep=' ') -> List[str]:
#         "Convert a list of `nums` to their tokens."
#         nums = np.array(nums).tolist()
#         return sep.join(
#             self.tokenizer.convert_ids_to_tokens(nums)
#         ) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)

#     def __getstate__(self):
#         return {'itos': self.itos, 'tokenizer': self.tokenizer}

#     def __setstate__(self, state: dict):
#         self.itos = state['itos']
#         self.tokenizer = state['tokenizer']
#         self.stoi = collections.defaultdict(
#             int, {v: k
#                   for k, v in enumerate(self.itos)})


class Model():
    """
    The Model class defines the Machine Learning model which should be used for training/evaluation/prediction
    """
    def __init__(self):
        super().__init__(
        )  # Inherit methods from the super class which this class extends from
        self.fai_tok = None

    def fit(self, review_sentiment_pairs):
        return

    def load_tokenizer(self):
        if self.fai_tok is None:
            self.ptt = XLNetTokenizer.from_pretrained('xlnet-large-cased')
            # FastAiTokenizer()
            self.fai_tok = Tokenizer(self.ptt)
        return self.ptt

    def tokenize(self, txt):
        self.load_tokenizer()
        # self.ptt.save_vocabulary()
        return self.ptt.tokenize(txt)

    def make_databunch(self):
        pass
    
    def forward(self, sample):
        return

    def save_model(self, save_path):
        with open(save_path, 'wb') as fp:
            pickle.dump(self, fp)

    def load_model(self, model_path):
        with open(model_path, 'rb') as fp:
            model = pickle.load(fp)
            self.__dict__.update(model.__dict__)

    def __call__(self, sample):
        return self.forward(sample)
