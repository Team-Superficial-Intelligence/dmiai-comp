from collections import defaultdict
import pickle
import json
from typing import List
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from fastai import *
from fastai.text.core import BaseTokenizer, Tokenizer
# from fastai.callbacks import *
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig


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

    def tokenize(self):
        if self.fai_tok is None:
            self.ptt = FastAiTokenizer()
            self.fai_tok = Tokenizer(self.ptt)
        self.ptt.save_vocabulary()

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
