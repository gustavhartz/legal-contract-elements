import pickle

import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset, dataset

nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner", "lemmatizer"])

# TODO: Make the DRQA dataset class the same as the BIDAF dataset class. The only difference is that the DRQA dataset class has 
#       has som options that should be set in the config.


class BiDAFDataset(Dataset):

    def __init__(self, pickle_file_path, char2idx_path):
        """Dataset for the BiDAF models

        Args:
            pickle_file_path ([type]): [description]
        """

        self.frame = pd.read_pickle(pickle_file_path)
        # Hardcoded fix for question length issues
        self.frame = self.frame[self.frame['id'] != "572fdefb947a6a140053cd8d"]

        self.max_context_len = max([len(ctx) for ctx in self.frame.context_ids])
        # Very slow to load
        # Using hardcoded values
        # self.max_word_len = max(self.frame.context.apply(lambda x: max([len(word.text) for word in nlp(x, disable=['parser', 'tagger', 'ner'])])))
        # self.max_question_len = max(self.frame.question.apply(lambda x: max([len(word.text) for word in nlp(x, disable=['parser', 'tagger', 'ner'])])))
        self.max_word_len = 38
        self.max_question_len = 27

        self.max_question_len = max([len(ques) for ques in self.frame.question_ids])
        with open(char2idx_path, 'rb') as handle:
            self.char2idx = pickle.load(handle)

    def get_span(self, text):

        text = nlp(text, disable=['parser', 'tagger', 'ner'])
        span = [(w.idx, w.idx + len(w.text)) for w in text]
        return span

    def __len__(self):
        return len(self.frame)

    def make_char_vector(self, max_sent_len, max_word_len, sentence):

        char_vec = torch.ones(max_sent_len, max_word_len).type(torch.LongTensor)

        for i, word in enumerate(nlp(sentence, disable=['parser', 'tagger', 'ner'])):
            for j, ch in enumerate(word.text):
                char_vec[i][j] = self.char2idx.get(ch, 0)

        return char_vec  

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        spans = []
        context_text = []
        answer_text = []
        batch = self.frame.iloc[idx]
        padded_context = torch.LongTensor(self.max_context_len).fill_(1)

        context_text.append(batch.context)
        spans.append(self.get_span(batch.context))

        # Calculate the char_ctx vector
        # Only for a single word - Should be fixed to work for a batch thus the init stuff
        max_word_ctx = max([len(word.text) for word in nlp(batch.context, disable=['parser', 'tagger', 'ner'])])
        char_ctx = self.make_char_vector(self.max_context_len, self.max_word_len, batch.context)

        # Calculate the char_ques vector
        # Only for a single question - Should be fixed to work for a batch thus the init stuff
        max_word_ques = max([len(word.text) for word in nlp(batch.question, disable=['parser', 'tagger', 'ner'])])
        char_ques = self.make_char_vector(self.max_question_len, self.max_question_len, batch.question)

        answer_text.append(batch.answer)

        padded_context[:len(batch.context_ids)] = torch.LongTensor(batch.context_ids)

        padded_question = torch.LongTensor(self.max_question_len).fill_(1)

        padded_question[:len(batch.question_ids)] = torch.LongTensor(batch.question_ids)

        label = torch.LongTensor(batch.label_idx)

        # Not used in the model
        context_mask = torch.eq(padded_context, 1)
        question_mask = torch.eq(padded_question, 1)

        ids = batch.id[0]  
        return (padded_context, padded_question, char_ctx, char_ques, label, context_text, answer_text, ids)


if __name__ == "__main__":
    # For shuffle dataloader
    g = torch.Generator()
    g.manual_seed(10)
    import os
    print(os.getcwd())
    train_dataset = BiDAFDataset("./data/processed/squad_bidaf/bidaftrain.pkl", "./data/processed/squad_bidaf/qanetc2id.pickle")
    print(len(train_dataset))
    train_dataset.frame = train_dataset.frame[0:10]
    print(len(train_dataset))
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        generator=g,
    )
    for batch, data in enumerate(trainloader):
        print(batch)
        print(len(data[0]))
        a = data
        print(a[0].shape, a[1].shape, a[2].shape, a[3].shape, a[4].shape
              )
        break
