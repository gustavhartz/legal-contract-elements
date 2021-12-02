import json
import os
import pickle
import re
import string
from collections import Counter

import pytorch_lightning as pl
import torch
from bidaf import BiDAF
from bidaf_dataset import BiDAFDataset
from bidaf_lightning import bidafLightning
from pytorch_lightning.loggers import WandbLogger


def train():

    torch.manual_seed(10)

    # For shuffle dataloader
    g = torch.Generator()
    g.manual_seed(10)

    print("CWD: ", os.getcwd())

    train_set = BiDAFDataset(
        "../../../data/processed/squad_bidaf/bidaftrain.pkl",
        "../../../data/processed/squad_bidaf/qanetc2id.pickle"
    )
    val_set = BiDAFDataset(
        "../../../data/processed/squad_bidaf/bidafvalid.pkl",
        "../../../data/processed/squad_bidaf/qanetc2id.pickle"
    )
    # Reduce size for testing

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=False,
    )

    validloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=32,
        num_workers=6
    )
    with open("../../../data/processed/squad_bidaf/qanetc2id.pickle", 'rb') as file:
        char2idx = pickle.load(file)

    with open("../../../data/processed/squad_bidaf/qanetw2id.pickle", 'rb') as file:
        word2idx = pickle.load(file)
    idx2word = {v: k for k, v in word2idx.items()}

    evaluate_func = evaluate

    # Model
    CHAR_VOCAB_DIM = len(char2idx)
    del char2idx

    EMB_DIM = 100
    CHAR_EMB_DIM = 8
    NUM_OUTPUT_CHANNELS = 100
    KERNEL_SIZE = (8, 5)
    HIDDEN_DIM = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BiDAF(CHAR_VOCAB_DIM, 
                  EMB_DIM, 
                  CHAR_EMB_DIM, 
                  NUM_OUTPUT_CHANNELS, 
                  KERNEL_SIZE, 
                  HIDDEN_DIM, 
                  device,
                  "/Users/gustavhartz/Projects/School/legal-contract-elements/data/processed/squad_bidaf/bidafglove_tv.npy").to(device)

    litmodel = bidafLightning(model, idx2word=idx2word, device=device, evaluate_func=evaluate_func, optimizer_lr=0.002)
    wandb_logger = WandbLogger(project="legal-contract-analysis")
    trainer = pl.Trainer(max_epochs=5, gpus=0, logger=wandb_logger, profiler="simple")

    trainer.fit(litmodel, trainloader, validloader)


def evaluate(predictions, **kwargs):
    '''
    Gets a dictionary of predictions with question_id as key
    and prediction as value. The validation dataset has multiple 
    answers for a single question. Hence we compare our prediction
    with all the answers and choose the one that gives us
    the maximum metric (em or f1). 
    This method first parses the JSON file, gets all the answers
    for a given id and then passes the list of answers and the 
    predictions to calculate em, f1.


    :param dict predictions
    Returns
    : exact_match: 1 if the prediction and ground truth 
      match exactly, 0 otherwise.
    : f1_score: 
    '''

    with open('../../../data/raw/dev-v1.1.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    dataset = dataset['data']
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    continue

                ground_truths = list(map(lambda x: x['text'], qa['answers']))

                prediction = predictions[qa['id']]

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)

                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def evaluate_single(predictions, answers, **kwargs):
    '''
    Gets a dictionary of predictions with question_id as key
    and prediction as value. The validation dataset has multiple 
    answers for a single question. Hence we compare our prediction
    with all the answers and choose the one that gives us
    the maximum metric (em or f1). 
    This method first parses the JSON file, gets all the answers
    for a given id and then passes the list of answers and the 
    predictions to calculate em, f1.


    :param dict predictions
    Returns
    : exact_match: 1 if the prediction and ground truth 
      match exactly, 0 otherwise.
    : f1_score: 
    '''
    assert len(predictions) == len(answers)
    f1 = exact_match = total = 0
    for key, value in predictions.items():
        prediction = value
        ground_truths = [answers[key]]

        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    total = len(predictions)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def normalize_answer(s):
    '''
    Performs a series of cleaning steps on the ground truth and 
    predicted answer.
    '''
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    '''
    Returns maximum value of metrics for predicition by model against
    multiple ground truths.

    :param func metric_fn: can be 'exact_match_score' or 'f1_score'
    :param str prediction: predicted answer span by the model
    :param list ground_truths: list of ground truths against which
                               metrics are calculated. Maximum values of 
                               metrics are chosen.


    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    '''
    Returns f1 score of two strings.
    '''
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    '''
    Returns exact_match_score of two strings.
    '''
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def epoch_time(start_time, end_time):
    '''
    Helper function to record epoch time.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    train()
