import pandas as pd
import torch
from torch.utils.data import Dataset, dataset
import spacy
nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner", "lemmatizer"])


class DRQADataset(Dataset):

    def __init__(self, pickle_file_path):
        """Dataset for the DRQA models

        Args:
            pickle_file_path ([type]): [description]
        """

        self.frame = pd.read_pickle(pickle_file_path)
        self.max_context_len = max([len(ctx) for ctx in self.frame.context_ids])
        self.max_question_len = max([len(ques) for ques in self.frame.question_ids])

    def get_span(self, text):

        text = nlp(text, disable=['parser', 'tagger', 'ner'])
        span = [(w.idx, w.idx + len(w.text)) for w in text]
        return span

    def __len__(self):
        return len(self.frame)

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

        answer_text.append(batch.answer)

        padded_context[:len(batch.context_ids)] = torch.LongTensor(batch.context_ids)

        padded_question = torch.LongTensor(self.max_question_len).fill_(1)

        padded_question[:len(batch.question_ids)] = torch.LongTensor(batch.question_ids)

        label = torch.LongTensor(batch.label_idx)
        context_mask = torch.eq(padded_context, 1)
        question_mask = torch.eq(padded_question, 1)

        ids = list(batch.id)  
        return (padded_context, padded_question, context_mask, 
                question_mask, label, context_text, answer_text, ids)


if __name__ == "__main__":
    # For shuffle dataloader
    g = torch.Generator()
    g.manual_seed(10)
    import os
    print(os.getcwd())
    train_dataset = DRQADataset('./data/processed/squad_drqa/drqa_train.pkl')
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
