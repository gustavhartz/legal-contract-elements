# -*- coding: utf-8 -*-

"""Download and pre-process SQuAD and GloVe datasets
Usage:
    > source activate squad
    > python setup.py
Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py
    > https://github.com/chrischute/squad
Author:
    Gustav Hartz (s174315@student.dtu.dk)
"""

import json
import logging
import os
import pickle
import re
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from helpers_squad import (add_word_embedding, build_word_vocab,
                           gather_text_for_vocab)
from tqdm import tqdm

print("sf")
logger = logging.getLogger(__name__)
logger_spacy = logging.getLogger("spacy")
logger_spacy.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"\[W108\]", category=UserWarning)
nlp = spacy.load('en_core_web_sm')


def parse_data(path: str) -> list:
    '''
    Parses the JSON file of CUDA dataset
    '''
    data = []
    with open(path, encoding="utf-8") as f:
        cuad = json.load(f)
        # Contract
        for example in cuad["data"]:
            title = example.get("title", "").strip()
            # Paragraph in contract
            # We only look at the first one
            for paragraph in example["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    if qa.get("is_impossible"):
                        continue
                    question = qa["question"].strip()
                    id_ = qa["id"]

                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"].strip() for answer in qa["answers"]]
                    answer_end = [context.find(x) + len(x) for x in answers if context.find(x)]
                    if len(answers) > 1:
                        continue
                    if answer_end and len(context[min(answer_starts):max(answer_end)]) < 2000:
                        ctx_offset = max(1, min(answer_starts) - 200)
                        data.append((id_, question, answers[0], [answer_starts[0] - ctx_offset, answer_end[0] -
                                    ctx_offset], 
                                    context[ctx_offset:min(len(context), max(answer_end) + 500)], ctx_offset))

    return data


def __prepare_data_frame_from_cuda(file_path, filter_questions=True):
    data = parse_data(file_path)
    df = pd.DataFrame(data, columns=["id_", "question", "answer", "label", "context", "ctx_offset"])
    if filter_questions:
        df = df[df['question'].isin(['Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer. Details: The name of the contract',
                                     'Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract',
                                     """'Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer. Details: On what date will the contract\'s initial term expire?'""",
                                     'Highlight the parts (if any) of this contract related to "Renewal Term" that should be reviewed by a lawyer. Details: What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.',
                                     ])]
    return df


def data_frame_from_cuda(path: str, save_files=True, filter_questions=True):
    """Process the raw json files and add the various data embedding features to it

    Args:
        path (str): Path to cuda data set
    """
    global nlp
    nlp = spacy.load('en_core_web_sm')
    print(path)
    df = __prepare_data_frame_from_cuda(path, filter_questions)
    vocab_text = gather_text_for_vocab([df])
    print("Number of sentences in dataset: ", len(vocab_text))
    print("Building vocab")
    word2idx, idx2word, word_vocab = build_word_vocab(vocab_text)
    print("numericalize context and questions for training and validation set. Expected processing time 1 minute")
    df = add_word_embedding(df, word2idx, idx2word)
    root_path = str(Path(__file__).resolve().parents[2]) + "/data/processed/cuad_drqa/"
    df.to_pickle(root_path + "data.pkl")
    print("data done")
    print("Saving files")
    if save_files:
        with open(root_path + 'drqa_word2idx.pickle', 'wb+') as handle:
            pickle.dump(word2idx, handle)
        with open(root_path + 'drqa_idx2word.pickle', 'wb+') as handle:
            pickle.dump(idx2word, handle)
        with open(root_path + 'drqa_word_vocab.pickle', 'wb+') as handle:
            pickle.dump(word_vocab, handle)
