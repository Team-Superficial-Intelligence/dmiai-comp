import base64
from warnings import simplefilter
import numpy as np
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.config import global_args
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from tqdm.utils import SimpleTextIOWrapper
from transformers import RobertaTokenizer, RobertaTokenizerFast, GPT2TokenizerFast
from scipy.special import softmax
import time

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


class MyClassificationModel(ClassificationModel):
    def add_pad_token(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token


def load_model(mt='distilbert', pt_dir='outputs', tt='roberta'):
    tok = RobertaTokenizerFast
    if tt == 'gpt2':
        tok = GPT2TokenizerFast
    # global_args.global_args["use_cached_eval_features"] = True
    return ClassificationModel(mt, pt_dir, tokenizer_type=tok)


def load_data(src='imdb_sup'):
    ext = 'gz'
    if src == 'imdb_sup':
        ext = 'zip'
    fn = './data/{}.csv.{}'.format(src, ext)
    logging.log(logging.INFO, "Using file: {}".format(fn))
    df = pd.read_csv(fn, header=None, skiprows=1)
    logging.log(
        logging.INFO,
        "Imported {} rows, {} columns".format(df.shape[0], df.shape[1]))
    if src == 'imdb_sup':
        logging.log(logging.INFO, "Removing extra column...")
        del df[2]
        # ratings 5 and 6 were missing because sentiments
        logging.log(logging.INFO, "Filling gaps...")
        df.loc[df[1] > 6, 1] -= 2
        # index start from 0
    else:
        # 8 million is too many
        logging.log(logging.INFO, "Gathering random sample...")
        df = df.sample(n=100000)
    logging.log(logging.INFO, "Decrementing rating...")
    df[1] -= 1

    df.columns = ["text", "labels"]
    logging.log(
        logging.INFO,
        "Final shape: {} rows, {} columns".format(df.shape[0], df.shape[1]))

    return df


def model_config(m):
    conf_opts = {
        # OK - slow af ~ 3h, not completed
        'xlnet1': ['xlnet', 'xlnet-large-cased', 2, None],
        # untested
        'xlnet2': ['xlnet', 'xlnet-base-cased', 16, None],
        # OK - not bad ~ 1h / epoch, not completed
        'roberta1': ['roberta', 'roberta-large', 4, None],
        # untested
        'bert1': ['bert', 'bert-base-cased', 16, None],
        # might be worth trying
        # could give us an advantage on non-english
        'xlmr1': ['xlmroberta', 'xlm-roberta-base', 16, None],
        'xlmr2': ['xlmroberta', 'xlm-roberta-large', 16, None],
        # for a speed test?
        'dbert1':
        ['distilbert', 'distilbert-base-multilingual-cased', 16, None],
        # 20min epoch with 128 bs. looking good on loss
        'dbert2':
        ['distilbert', 'distilroberta-base', 128, RobertaTokenizerFast],
        'dbert3': ['distilbert', 'distilgpt2', 128, GPT2TokenizerFast],
    }
    return conf_opts[m]


def main(src='imdb_sup'):

    train_df, eval_df = train_test_split(load_data(src), train_size=0.5)
    # train_df, eval_df = [[('foo', 1)], [('bar', 2)]]
    use_conf = 'dbert3'
    m, pt, bs, tt = model_config(use_conf)
    if src == 'imdb_sup':
        n_labels = 8
        epochs = 5
    elif src == 'amazon':
        n_labels = 5
        epochs = 5
        #pt = 'outputs-distilbert2'
    # Optional model configuration
    # For xlnet large batch size max is 2
    model_args = ClassificationArgs(num_train_epochs=epochs,
                                    overwrite_output_dir=True,
                                    train_batch_size=bs)
    # model_args.weight_decay = 0.01
    model = MyClassificationModel(m,
                                  pt,
                                  num_labels=n_labels,
                                  use_cuda=True,
                                  args=model_args,
                                  tokenizer_type=tt)
    model.add_pad_token()
    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    # logging.log(logging.INFO, str(result))
    # logging.log(logging.INFO, str(model_outputs))
    # logging.log(logging.INFO, str(wrong_predictions))


def train_more():
    use_conf = 'dbert2'
    m, pt, bs, tt = model_config(use_conf)

    model_args = ClassificationArgs(num_train_epochs=5,
                                    overwrite_output_dir=True,
                                    train_batch_size=bs)

    model = ClassificationModel("distilbert",
                                "outputs-distilbert2",
                                tokenizer_type=RobertaTokenizerFast)
    train_df, eval_df = train_test_split(load_data(), train_size=0.5)


def eval_local():
    train_df, eval_df = train_test_split(load_data())
    model = ClassificationModel("xlnet", "outputs/checkpoint-18750-epoch-1")
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    logging.log(logging.INFO, str(result))
    logging.log(logging.INFO, str(model_outputs))
    logging.log(logging.INFO, str(wrong_predictions))


def scale_predict(a1d, scale):

    return round(round(np.dot(scale, a1d) / 2, 1) * 2) / 2


def raw_output_to_stars(raw_outputs, scale=None):
    if scale is None:
        scale = np.array([1, 2, 3, 4, 7, 8, 9, 10])
    probabilities = softmax(raw_outputs, axis=1)
    probs_altered = np.apply_along_axis(scale_predict,
                                        arr=probabilities,
                                        axis=1,
                                        scale=scale)
    return probs_altered.tolist()


def predict_stars(model, to_predict):
    _, raw_outputs = model.predict(to_predict)
    return raw_output_to_stars(raw_outputs)


def test_predictions():
    model = ClassificationModel("distilbert",
                                "outputs-distil-gpt2",
                                tokenizer_type=GPT2TokenizerFast)
    preds, raw_outputs = model.predict([
        "This is a shit movie",
        "this is a great movie",
    ])
    tstart = time.time()
    preds, raw_outputs = model.predict([
        "This is a shit movie",
        "this is a great movie",
    ])
    tend = time.time()
    stars = raw_output_to_stars(raw_outputs)
    logging.log(logging.INFO, "stars:" + str(stars))
    logging.log(logging.INFO, "time for 2 predictions:" + str(tend - tstart))


if __name__ == "__main__":
    # test_predictions()
    main()
