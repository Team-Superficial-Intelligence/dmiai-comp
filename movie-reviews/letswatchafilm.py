import base64
from warnings import simplefilter
import numpy as np
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.config import global_args
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from tqdm.utils import SimpleTextIOWrapper
from transformers import RobertaTokenizer, RobertaTokenizerFast
from scipy.special import softmax
import time

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def load_model():
    # global_args.global_args["use_cached_eval_features"] = True
    return ClassificationModel("distilbert",
                               "outputs",
                               tokenizer_type=RobertaTokenizerFast)


def load_data(src='imdb_sup'):
    ext = 'gz'
    if src == 'imdb_sup':
        ext = 'zip'
    df = pd.read_csv('./data/{}.csv.{}'.format(src, ext),
                     header=None,
                     skiprows=1)
    if src == 'imdb_sup':
        del df[2]
        # ratings 5 and 6 were missing because sentiments
        df.loc[df[1] > 6, 1] -= 2
        # index start from 0

    df[1] -= 1

    df.columns = ["text", "labels"]

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
        'dbert3': ['distilbert', 'distilgpt2', 16, None],
    }
    return conf_opts[m]


def main(src='imdb_sup'):

    train_df, eval_df = train_test_split(load_data(src), train_size=0.5)
    use_conf = 'dbert2'
    m, pt, bs, tt = model_config(use_conf)
    if src == 'imdb_sup':
        n_labels = 8
        epochs = 5
    elif src == 'amazon':
        n_labels = 5
        epochs = 3
    # Optional model configuration
    # For xlnet large batch size max is 2
    model_args = ClassificationArgs(num_train_epochs=epochs,
                                    overwrite_output_dir=True,
                                    train_batch_size=bs)
    # model_args.weight_decay = 0.01
    model = ClassificationModel(m,
                                pt,
                                num_labels=n_labels,
                                use_cuda=True,
                                args=model_args,
                                tokenizer_type=tt)

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


def scale_predict(a1d):
    scale = np.array([1, 2, 3, 4, 7, 8, 9, 10])
    return round(round(np.dot(scale, a1d) / 2, 1) * 2) / 2


def raw_output_to_stars(raw_outputs):
    probabilities = softmax(raw_outputs, axis=1)
    probs_altered = np.apply_along_axis(scale_predict,
                                        arr=probabilities,
                                        axis=1)
    return probs_altered.tolist()


def predict_stars(model, to_predict):
    _, raw_outputs = model.predict(to_predict)
    return raw_output_to_stars(raw_outputs)


def test_predictions():
    model = ClassificationModel("distilbert",
                                "outputs-distilbert2",
                                tokenizer_type=RobertaTokenizerFast)
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
    logging.log(logging.INFO, "time for 2 predictions:" + str(tend - tstart))


if __name__ == "__main__":
    # test_predictions()
    main('amazon')