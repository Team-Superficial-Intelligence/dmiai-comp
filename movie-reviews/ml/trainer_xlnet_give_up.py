from fastai.text.all import *
from fastai.learner import Transform
from fastai.learner import Learner
from torch import Tensor
from transformers import PreTrainedTokenizer, XLNetTokenizerFast, XLNetConfig, XLNetForSequenceClassification, AdamW
from functools import partial
from fastai.callback import *
from ml.model import Model
import pandas as pd
import re
from tqdm import tqdm  # Wraps iterables and prints a progress bar
import os
import pickle

tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-large-cased')


def tokenize(text):
    toks = tokenizer.tokenize(text)
    # return tensor(tokenizer.convert_tokens_to_ids(toks))
    return tensor(tokenizer.convert_tokens_to_ids(toks))


class DropOutput(Callback):
    def after_pred(self):
        self.learn.pred = self.pred[0]


class BadTransformersTokenizer(Transform):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encodes(self, x):
        return x if isinstance(x, Tensor) else tokenize(x)

    def decodes(self, x):
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


class TransformersTokenizer(Transform):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encodes(self, x):
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))

    def decodes(self, x):
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


class Trainer:
    """
    The Trainer class is used for training a model instance based on the Model class found in ml.model.py.
    In order to get started with training a model the following steps needs to be taken:
    1. Define the Model class in ml.model.py
    2. Prepare train data on which the model should be trained with by implementing the _read_train_data() function and
    the _preprocess_train_data() function
    """
    def __init__(self):
        # creates an instance of the Model class (see guidelines in ml.model.py)
        self.model = Model()

    def train2(self):
        from fast_bert.data_cls import BertDataBunch
        from fast_bert.learner_cls import BertLearner
        from fast_bert.metrics import accuracy
        import logging

        databunch = BertDataBunch(
            './data/kaggle/',
            './data/kaggle/',
            tokenizer='xlnet-base-cased',
            train_file='train.csv',
            val_file='test.csv',
            text_col='text',
            label_col=['zero', 'one', 'two', 'three', 'four'],
            batch_size_per_gpu=4,
            max_seq_length=256,
            multi_gpu=False,
            multi_label=True,
            model_type='xlnet')

        logger = logging.getLogger()
        device_cuda = torch.device("cuda")
        metrics = [{'name': 'accuracy', 'function': accuracy}]

        learner = BertLearner.from_pretrained_model(
            databunch,
            pretrained_path='xlnet-base-cased',
            metrics=metrics,
            device=device_cuda,
            logger=logger,
            output_dir='./data/out',
            finetuned_wgts_path=None,
            warmup_steps=500,
            multi_gpu=False,
            is_fp16=True,
            multi_label=True,
            logging_steps=50)

        learner.lr_find(start_lr=1e-5, optimizer_type='lamb')
        learner.fit(
            epochs=6,
            lr=6e-5,
            validate=True,  # Evaluate the model after each epoch
            schedule_type="warmup_cosine",
            optimizer_type="lamb")

        # dataset_path_train = '/workspace/data/kaggle/train.tsv.zip'
        # dataset_path_test = '/workspace/data/kaggle/test.tsv.zip'
        # df_train = self._load_train_data(dataset_path_train)
        # df_valid = self._load_train_data(dataset_path_test)

    def train(self):
        """
        Starts the training of a model based on data loaded by the self._load_train_data function
        """

        # Unpack request
        #dataset_path_train = './data/kaggle/train.tsv.zip'
        #dataset_path_test = './data/kaggle/test.tsv.zip'
        dataset_path_train = './data/kaggle/train.tsv.zip'
        dataset_path_test = './data/kaggle/test.tsv.zip'
        df_train = self._load_train_data(dataset_path_train)
        df_valid = self._load_train_data(dataset_path_test)
        all_texts = np.concatenate([df_train[2].values, df_valid[2].values])

        splits = [
            range_of(df_train),
            list(range(len(df_train), len(all_texts)))
        ]
        tls = TfmdLists(all_texts,
                        TransformersTokenizer(tokenizer),
                        splits=splits,
                        dl_type=LMDataLoader)
        bs, sl = 16, 256
        dls = tls.dataloaders(bs=bs, seq_len=sl)
        config = XLNetConfig.from_pretrained('xlnet-large-cased')
        print(config)
        config.num_labels = 5
        # config.d_inner = bs
        config.use_bfloat16 = True
        transformer_model = XLNetForSequenceClassification.from_pretrained(
            'xlnet-large-cased', config=config)

        optAdamW = partial(AdamW, correct_bias=False)
        mlearner = Learner(dls,
                           transformer_model,
                           opt_func=optAdamW,
                           loss_func=CrossEntropyLossFlat(),
                           cbs=[DropOutput],
                           metrics=Perplexity()).to_fp16()
        mlearner.validate()
        mlearner.save('untrain')
        mlearner.load('untrain')
        mlearner.freeze_to(-1)
        mlearner.summary()
        return "hehehehe"

    def _load_train_data(self, dataset_path):
        return pd.read_csv(dataset_path, sep="\t", header=None, skiprows=1)

    def __call__(self, request):
        return self.train(request)