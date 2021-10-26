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
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger_spacy = logging.getLogger("spacy")
logger_spacy.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"\[W108\]", category=UserWarning)
nlp = spacy.load('en_core_web_sm')


def load_json(path):
    '''
    Loads the JSON file of the Squad dataset.
    Returns the json object of the dataset.
    '''
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Length of data: ", len(data['data']))
    print("Data Keys: ", data['data'][0].keys())
    print("Title: ", data['data'][0]['title'])

    return data


def parse_data(data: dict) -> list:
    '''
    Parses the JSON file of Squad dataset by looping through the
    keys and values and returns a list of dictionaries with
    context, query and label triplets being the keys of each dict.
    '''
    data = data['data']
    qa_list = []
    logger.info("Parsing data")
    for paragraphs in tqdm(data):

        for para in paragraphs['paragraphs']:
            context = para['context']

            for qa in para['qas']:
                id = qa['id']
                question = qa['question']
                for ans in qa['answers']:
                    answer = ans['text']
                    ans_start = ans['answer_start']
                    ans_end = ans_start + len(answer)
                    qa_dict = {}
                    qa_dict['id'] = id
                    qa_dict['context'] = context
                    qa_dict['question'] = question
                    qa_dict['label'] = [ans_start, ans_end]

                    qa_dict['answer'] = answer
                    qa_list.append(qa_dict)

    return qa_list


def filter_large_examples(df):
    '''
    Returns ids of examples where context lengths, query lengths and answer lengths are
    above a particular threshold. These ids can then be dropped from the dataframe. 
    This is explicitly mentioned in QANet but can be done for other models as well.
    '''

    ctx_lens = []
    query_lens = []
    ans_lens = []
    for index, row in df.iterrows():
        ctx_tokens = [w.text for w in nlp(row.context, disable=['parser', 'ner', 'tagger'])]
        if len(ctx_tokens) > 400:
            ctx_lens.append(row.name)

        query_tokens = [w.text for w in nlp(row.question, disable=['parser', 'tagger', 'ner'])]
        if len(query_tokens) > 50:
            query_lens.append(row.name)

        ans_tokens = [w.text for w in nlp(row.answer, disable=['parser', 'tagger', 'ner'])]
        if len(ans_tokens) > 30:
            ans_lens.append(row.name)

        assert row.name == index

    return set(ans_lens + ctx_lens + query_lens)


def gather_text_for_vocab(dfs: list):
    '''
    Gathers text from contexts and questions to build a vocabulary.

    :param dfs: list of dataframes of SQUAD dataset.
    :returns: list of contexts and questions
    '''

    text = []
    total = 0
    for df in dfs:
        unique_contexts = list(df.context.unique())
        unique_questions = list(df.question.unique())
        total += df.context.nunique() + df.question.nunique()
        text.extend(unique_contexts + unique_questions)

    assert len(text) == total

    return text


def build_word_vocab(vocab_text):
    '''
    Builds a word-level vocabulary from the given text.

    :param list vocab_text: list of contexts and questions
    :returns 
        dict word2idx: word to index mapping of words
        dict idx2word: integer to word mapping
        list word_vocab: list of words sorted by frequency
    '''

    words = []
    for sent in vocab_text:
        for word in nlp(sent, disable=['parser', 'tagger', 'ner']):
            words.append(word.text)

    word_counter = Counter(words)
    word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    print(f"raw-vocab: {len(word_vocab)}")
    word_vocab.insert(0, '<unk>')
    word_vocab.insert(1, '<pad>')
    print(f"vocab-length: {len(word_vocab)}")
    word2idx = {word: idx for idx, word in enumerate(word_vocab)}
    print(f"word2idx-length: {len(word2idx)}")
    idx2word = {v: k for k, v in word2idx.items()}

    return word2idx, idx2word, word_vocab


def build_char_vocab(vocab_text):
    '''
    Builds a character-level vocabulary from the given text.

    :param list vocab_text: list of contexts and questions
    :returns
        dict char2idx: character to index mapping of words
        list char_vocab: list of characters sorted by frequency
    '''

    chars = []
    for sent in vocab_text:
        for ch in sent:
            chars.append(ch)

    char_counter = Counter(chars)
    char_vocab = sorted(char_counter, key=char_counter.get, reverse=True)
    print(f"raw-char-vocab: {len(char_vocab)}")
    high_freq_char = [char for char, count in char_counter.items() if count >= 20]
    char_vocab = list(set(char_vocab).intersection(set(high_freq_char)))
    print(f"char-vocab-intersect: {len(char_vocab)}")
    char_vocab.insert(0, '<unk>')
    char_vocab.insert(1, '<pad>')
    char2idx = {char: idx for idx, char in enumerate(char_vocab)}
    print(f"char2idx-length: {len(char2idx)}")

    return char2idx, char_vocab


def context_to_ids(text, word2idx):
    '''
    Converts context text to their respective ids by mapping each word
    using word2idx. Input text is tokenized using spacy tokenizer first.

    :param str text: context text to be converted
    :param dict word2idx: word to id mapping

    :returns list context_ids: list of mapped ids

    :raises assertion error: sanity check

    '''

    context_tokens = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
    context_ids = [word2idx[word] for word in context_tokens]

    assert len(context_ids) == len(context_tokens)
    return context_ids


def question_to_ids(text, word2idx):
    '''
    Converts question text to their respective ids by mapping each word
    using word2idx. Input text is tokenized using spacy tokenizer first.

    :param str text: question text to be converted
    :param dict word2idx: word to id mapping
    :returns list context_ids: list of mapped ids

    :raises assertion error: sanity check

    '''

    question_tokens = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
    question_ids = [word2idx[word] for word in question_tokens]

    assert len(question_ids) == len(question_tokens)
    return question_ids


def test_indices(df, idx2word):
    '''
    Performs the tests mentioned above. This method also gets the start and end of the answers
    with respect to the context_ids for each example.

    :param dataframe df: SQUAD df
    :param dict idx2word: inverse mapping of token ids to words
    :returns
        list start_value_error: example idx where the start idx is not found in the start spans
                                of the text
        list end_value_error: example idx where the end idx is not found in the end spans
                              of the text
        list assert_error: examples that fail assertion errors. A majority are due to the above errors

    '''

    start_value_error = []
    end_value_error = []
    assert_error = []
    for index, row in df.iterrows():

        answer_tokens = [w.text for w in nlp(row['answer'], disable=['parser', 'tagger', 'ner'])]

        start_token = answer_tokens[0]
        end_token = answer_tokens[-1]

        context_span = [(word.idx, word.idx + len(word.text))
                        for word in nlp(row['context'], disable=['parser', 'tagger', 'ner'])]

        if not context_span:
            assert_error.append(index)
            continue

        starts, ends = zip(*context_span)

        answer_start, answer_end = row['label']

        try:
            start_idx = starts.index(answer_start)
        except Exception:
            start_value_error.append(index)
        try:
            end_idx = ends.index(answer_end)
        except Exception:
            end_value_error.append(index)

        try:
            assert idx2word[row['context_ids'][start_idx]] == answer_tokens[0]
            assert idx2word[row['context_ids'][end_idx]] == answer_tokens[-1]
        except Exception:
            assert_error.append(index)

    return start_value_error, end_value_error, assert_error


def get_error_indices(df, idx2word):

    start_value_error, end_value_error, assert_error = test_indices(df, idx2word)
    err_idx = start_value_error + end_value_error + assert_error
    err_idx = set(err_idx)
    print(f"Number of error indices: {len(err_idx)}")

    return err_idx


def index_answer(row, idx2word):
    '''
    Takes in a row of the dataframe or one training example and
    returns a tuple of start and end positions of answer by calculating 
    spans.
    '''

    context_span = [(word.idx, word.idx + len(word.text))
                    for word in nlp(row.context, disable=['parser', 'tagger', 'ner'])]
    starts, ends = zip(*context_span)

    answer_start, answer_end = row.label
    start_idx = starts.index(answer_start)

    end_idx = ends.index(answer_end)

    ans_toks = [w.text for w in nlp(row.answer, disable=['parser', 'tagger', 'ner'])]
    ans_start = ans_toks[0]
    ans_end = ans_toks[-1]
    assert idx2word[row.context_ids[start_idx]] == ans_start
    assert idx2word[row.context_ids[end_idx]] == ans_end

    return [start_idx, end_idx]


def normalize_spaces(text):
    '''
    Removes extra white spaces from the context.
    '''
    text = re.sub(r'\s', ' ', text)
    return text


def __prepare_data_frame_from_squad(file_path):
    data = load_json(file_path)
    list = parse_data(data)
    df = pd.DataFrame(list)
    df.context = df.context.apply(normalize_spaces)
    return df


def add_word_embedding(df, word2idx, idx2word):
    df['context_ids'] = df.context.apply(context_to_ids, word2idx=word2idx)
    df['question_ids'] = df.question.apply(question_to_ids, word2idx=word2idx)
    err = get_error_indices(df, idx2word)
    df.drop(err, inplace=True)
    df_label_idx = df.apply(index_answer, axis=1, idx2word=idx2word)
    df['label_idx'] = df_label_idx
    return df


def data_frame_from_squad(train_path: str, valid_path: str, save_files=True):
    """ Process the raw json files and add the various data embedding features to it

    Args:
        train_path (str): Path to downloaded SQUAD train v.2
        valid_path (str): Path to downloaded SQUAD valid v.2
        save_files (bool, optional): If true it will save the dataframes to the processed data file. Defaults to True.
    """
    global nlp
    nlp = spacy.load('en_core_web_sm')
    print(train_path)
    train_df = __prepare_data_frame_from_squad(train_path)
    valid_df = __prepare_data_frame_from_squad(valid_path)
    vocab_text = gather_text_for_vocab([train_df, valid_df])
    print("Number of sentences in dataset: ", len(vocab_text))
    print("Building vocab")
    word2idx, idx2word, word_vocab = build_word_vocab(vocab_text)
    print("numericalize context and questions for training and validation set. Expected processing time 5 minutes")

    train_df = add_word_embedding(train_df, word2idx, idx2word)
    root_path = str(Path(__file__).resolve().parents[2]) + "/data/processed/squad_drqa/"
    train_df.to_pickle(root_path + "draq_train.pkl")
    del train_df
    print("Train done")
    valid_df = add_word_embedding(valid_df, word2idx, idx2word)
    valid_df.to_pickle(root_path + "draq_valid.pkl")
    print("Valid done")
    print("Saving files")
    if save_files:
        with open(root_path + 'drqa_word2idx.pickle', 'wb+') as handle:
            pickle.dump(word2idx, handle)
        with open(root_path + 'drqa_idx2word.pickle', 'wb+') as handle:
            pickle.dump(idx2word, handle)
        with open(root_path + 'drqa_word_vocab.pickle', 'wb+') as handle:
            pickle.dump(word_vocab, handle)
