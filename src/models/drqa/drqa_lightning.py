from collections import ChainMap

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class drqaLightning(pl.LightningModule):

    def __init__(self, model, optimizer_lr=0.01, optimizer_momentum=0.9, device=None, idx2word=None, evaluate_func=None):
        super().__init__()
        self.model = model
        self.lr = optimizer_lr
        self.momentum = optimizer_momentum
        self.device_ = device
        if not idx2word:
            raise NotImplementedError("idx2word not defined")
        if not evaluate_func:
            raise NotImplementedError("Evaluation Function for validation not defined")
        self.idx2word = idx2word
        self.evaluate_func = evaluate_func

    def training_step(self, batch, batch_idx):
        context, question, context_mask, question_mask, label, ctx, ans, ids = batch

        # place the tensors on GPU
        if self.device:
            context, question, context_mask, question_mask, label, ctx, ans, ids = context.to(self.device), 
            question.to(self.device), context_mask.to(self.device), question_mask.to(self.device), 
            label.to(self.device), ctx, ans, ids, preds = self.model(context, question, context_mask, question_mask)

        # forward pass, get the predictions
        start_pred, end_pred = preds
        # separate labels for start and end position
        start_label, end_label = label[:, 0], label[:, 1]
        # calculate loss
        loss = F.cross_entropy(start_pred, start_label) + F.cross_entropy(end_pred, end_label)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        context, question, context_mask, question_mask, label, ctx, ans, ids = batch

        if self.device:
            context, question, context_mask, question_mask, label, ctx, ans, ids = context.to(self.device), 
            question.to(self.device), context_mask.to(self.device), question_mask.to(self.device), 
            label.to(self.device), ctx, ans, ids
        # place the tensors on GPU

        preds = self.model(context, question, context_mask, question_mask)

        p1, p2 = preds

        # for preds
        batch_size, c_len = p1.size()

        y1, y2 = label[:, 0], label[:, 1]
        if self.device:
            p1 = p1.to('cpu')
            p2 = p2.to('cpu')
            y1 = y1.to('cpu')
            y2 = y2.to('cpu')
        loss = F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)

        self.log(
            "val_loss",
            loss.item(),
            on_epoch=True
        )

        predictions = {}
        print("_____")
        print(batch_size)
        print(len(ids))
        ls = nn.LogSoftmax(dim=1)
        mask = (torch.ones(c_len, c_len) * float('-inf')).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)

        score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
        score, s_idx = score.max(dim=1)
        score, e_idx = score.max(dim=1)
        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

        # stack predictions
        for i in range(batch_size):
            id = ids[i]
            pred = context[i][s_idx[i]:e_idx[i] + 1]
            pred = ' '.join([self.idx2word[idx.item()] for idx in pred])
            predictions[id] = pred

        return predictions

    def validation_epoch_end(self, validation_step_outputs):
        # Unpack dicts
        predictions = dict(ChainMap(*validation_step_outputs))

        em, f1 = self.evaluate_func(predictions)
        self.log(
            "val_em",
            em,
            on_epoch=True
        )
        self.log(
            "val_f1",
            f1,
            on_epoch=True
        )
        return f1

    def configure_optimizers(self):
        return torch.optim.Adamax(
            self.model.parameters(), lr=self.lr
        )
