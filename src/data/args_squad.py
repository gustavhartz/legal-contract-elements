# -*- coding: utf-8 -*-
"""
Code adapted from:
    > https://github.com/chrischute/squad
"""

import argparse


def get_setup_args():
    """For processing the squad dataset"""

    parser = argparse.ArgumentParser('Download and pre-process SQuAD')

    parser.add_argument('--train_url',
                        type=str,
                        default='https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json')
    parser.add_argument('--dev_url',
                        type=str,
                        default='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json')
    parser.add_argument('--glove_url',
                        type=str,
                        default='http://nlp.stanford.edu/data/glove.840B.300d.zip')
    parser.add_argument('--para_limit',
                        type=int,
                        default=400,
                        help='Max number of words in a paragraph')
    parser.add_argument('--ques_limit',
                        type=int,
                        default=50,
                        help='Max number of words to keep from a question')
    parser.add_argument('--test_para_limit',
                        type=int,
                        default=1000,
                        help='Max number of words in a paragraph at test time')
    parser.add_argument('--test_ques_limit',
                        type=int,
                        default=100,
                        help='Max number of words in a question at test time')
    parser.add_argument('--char_dim',
                        type=int,
                        default=64,
                        help='Size of char vectors (char-level embeddings)')
    parser.add_argument('--glove_dim',
                        type=int,
                        default=300,
                        help='Size of GloVe word vectors to use')
    parser.add_argument('--ans_limit',
                        type=int,
                        default=30,
                        help='Max number of words in a training example answer')
    parser.add_argument('--char_limit',
                        type=int,
                        default=16,
                        help='Max number of chars to keep from a word')

    args = parser.parse_args()

    return args
