from collections import ChainMap

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class bidafLightning(pl.LightningModule):

    def __init__(self, model, optimizer_lr=0.01, device=None, idx2word=None, evaluate_func=None):
        super().__init__()
        self.model = model
        self.lr = optimizer_lr
        self.device_ = device
        if device:
            model = model.to(device)
        if not idx2word:
            raise NotImplementedError("idx2word not defined")
        if not evaluate_func:
            raise NotImplementedError("Evaluation Function for validation not defined")
        self.idx2word = idx2word
        self.evaluate_func = evaluate_func

    def training_step(self, batch, batch_idx):
        context, question, char_ctx, char_ques, label, ctx_text, answers, ids = batch

        # place the tensors on GPU if not already there
        if self.device and not context.is_cuda:
            context = context.to(self.device) 
            question = question.to(self.device) 
            char_ctx = char_ctx.to(self.device)
            char_ques = char_ques.to(self.device) 
            label = label.to(self.device) 
        preds = self.model(context, question, char_ctx, char_ques)

        # forward pass, get the predictions
        start_pred, end_pred = preds
        # separate labels for start and end position
        start_label, end_label = label[:, 0], label[:, 1]
        # calculate loss
        loss = F.cross_entropy(start_pred, start_label) + F.cross_entropy(end_pred, end_label)

        self.log('loss', loss.item(), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        context, question, char_ctx, char_ques, label, ctx, ans, ids = batch
        # place the tensors on GPU if not already there
        if self.device and not context.is_cuda:
            context = context.to(self.device) 
            question = question.to(self.device) 
            char_ctx = char_ctx.to(self.device)
            char_ques = char_ques.to(self.device) 
            label = label.to(self.device) 

        preds = self.model(context, question, char_ctx, char_ques)

        p1, p2 = preds

        # for preds
        batch_size, c_len = p1.size()

        y1, y2 = label[:, 0], label[:, 1]

        # Maybe dont send to cpu yet
        if self.device:
            p1 = p1.to('cpu')
            p2 = p2.to('cpu')
            y1 = y1.to('cpu')
            y2 = y2.to('cpu')
        loss = F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)

        self.log('valid_loss', loss, on_step=True, on_epoch=True)

        predictions = {}
        answers = {}
        ls = nn.LogSoftmax(dim=1)
        mask = (torch.ones(c_len, c_len) * float('-inf')).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)

        score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
        score, s_idx = score.max(dim=1)
        score, e_idx = score.max(dim=1)
        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

        # unpack ans
        ans = ans[0]
        # stack predictions

        for i in range(batch_size):
            id = ids[i]
            pred = context[i][s_idx[i]:e_idx[i] + 1]
            pred = ' '.join([self.idx2word[idx.item()] for idx in pred])
            predictions[id] = pred
            answers[id] = ans[i]

        return (predictions, answers)

    def training_epoch_end(self, training_step_outputs):
        loss = [x['loss'].item() for x in training_step_outputs]
        self.log('avg_epoch_loss', sum(loss) / len(loss))

    def validation_epoch_end(self, validation_step_outputs):
        # Unpack dicts
        predictions = dict(ChainMap(*[x[0] for x in validation_step_outputs]))
        answers = dict(ChainMap(*[x[1] for x in validation_step_outputs]))

        em, f1 = self.evaluate_func(predictions, answers=answers)
        print(em)
        print(f1)
        self.log("val_em", em)
        self.log("val_f1", f1)

        return f1

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.model.parameters())
