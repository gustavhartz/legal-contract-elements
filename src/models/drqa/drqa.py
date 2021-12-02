"""Download and pre-process drqa model for Q&A
Code adapted from:
    > https://github.com/kushalj001/pytorch-question-answering
Author:
    Gustav Hartz (s174315@student.dtu.dk)
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchtext
from torch import nn

# Modules


class AlignQuestionEmbedding(nn.Module):

    def __init__(self, input_dim):        

        super().__init__()

        self.linear = nn.Linear(input_dim, input_dim)

        self.relu = nn.ReLU()

    def forward(self, context, question, question_mask):
        ctx_ = self.linear(context)
        ctx_ = self.relu(ctx_)
        qtn_ = self.linear(question)
        qtn_ = self.relu(qtn_)

        qtn_transpose = qtn_.permute(0, 2, 1)

        align_scores = torch.bmm(ctx_, qtn_transpose)

        qtn_mask = question_mask.unsqueeze(1).expand(align_scores.size())

        align_scores = align_scores.masked_fill(qtn_mask == 1, -float('inf'))

        align_scores_flat = align_scores.view(-1, question.size(1))

        alpha = F.softmax(align_scores_flat, dim=1)
        alpha = alpha.view(-1, context.shape[1], question.shape[1])

        align_embedding = torch.bmm(alpha, question)

        return align_embedding


class StackedBiLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):

        super().__init__()

        self.dropout = dropout

        self.num_layers = num_layers

        self.lstms = nn.ModuleList()

        for i in range(self.num_layers):

            input_dim = input_dim if i == 0 else hidden_dim * 2

            self.lstms.append(nn.LSTM(input_dim, hidden_dim,
                                      batch_first=True, bidirectional=True))

    def forward(self, x):

        outputs = [x]
        for i in range(self.num_layers):

            lstm_input = outputs[-1]
            lstm_out = F.dropout(lstm_input, p=self.dropout)
            lstm_out, (hidden, cell) = self.lstms[i](lstm_input)

            outputs.append(lstm_out)

        output = torch.cat(outputs[1:], dim=2)

        output = F.dropout(output, p=self.dropout)

        return output


class LinearAttentionLayer(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, question, question_mask):

        qtn = question.view(-1, question.shape[-1])

        attn_scores = self.linear(qtn)

        attn_scores = attn_scores.view(question.shape[0], question.shape[1])

        attn_scores = attn_scores.masked_fill(question_mask == 1, -float('inf'))

        alpha = F.softmax(attn_scores, dim=1)

        return alpha


def weighted_average(x, weights):

    weights = weights.unsqueeze(1)

    w = weights.bmm(x).squeeze(1)

    return w


class BilinearAttentionLayer(nn.Module):

    def __init__(self, context_dim, question_dim):

        super().__init__()

        self.linear = nn.Linear(question_dim, context_dim)

    def forward(self, context, question, context_mask):

        qtn_proj = self.linear(question)

        qtn_proj = qtn_proj.unsqueeze(2)

        scores = context.bmm(qtn_proj)

        scores = scores.squeeze(2)

        scores = scores.masked_fill(context_mask == 1, -float('inf'))

        return scores


class DocumentReader(nn.Module):

    def __init__(self, hidden_dim, embedding_dim, num_layers, num_directions, dropout, device, glove_matrix_path):

        super().__init__()

        self.device_ = device

        self.context_bilstm = StackedBiLSTM(embedding_dim * 2, hidden_dim, num_layers, dropout)

        self.question_bilstm = StackedBiLSTM(embedding_dim, hidden_dim, num_layers, dropout)

        self.glove_embedding = self.get_glove_embedding(glove_matrix_path)

        def tune_embedding(grad, words=1000):
            grad[words:] = 0
            return grad

        self.glove_embedding.weight.register_hook(tune_embedding)

        self.align_embedding = AlignQuestionEmbedding(embedding_dim)

        self.linear_attn_question = LinearAttentionLayer(hidden_dim * num_layers * num_directions) 

        self.bilinear_attn_start = BilinearAttentionLayer(hidden_dim * num_layers * num_directions, 
                                                          hidden_dim * num_layers * num_directions)

        self.bilinear_attn_end = BilinearAttentionLayer(hidden_dim * num_layers * num_directions,
                                                        hidden_dim * num_layers * num_directions)

        self.dropout = nn.Dropout(dropout)

    def get_glove_embedding(self, glove_matrix_path):

        weights_matrix = np.load(glove_matrix_path)
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix), freeze=False)

        return embedding

    def forward(self, context, question, context_mask, question_mask):

        ctx_embed = self.glove_embedding(context)

        ques_embed = self.glove_embedding(question)

        ctx_embed = self.dropout(ctx_embed)

        ques_embed = self.dropout(ques_embed)

        align_embed = self.align_embedding(ctx_embed, ques_embed, question_mask)

        ctx_bilstm_input = torch.cat([ctx_embed, align_embed], dim=2)

        ctx_outputs = self.context_bilstm(ctx_bilstm_input)

        qtn_outputs = self.question_bilstm(ques_embed)

        qtn_weights = self.linear_attn_question(qtn_outputs, question_mask)

        qtn_weighted = weighted_average(qtn_outputs, qtn_weights)

        start_scores = self.bilinear_attn_start(ctx_outputs, qtn_weighted, context_mask)

        end_scores = self.bilinear_attn_end(ctx_outputs, qtn_weighted, context_mask)

        return start_scores, end_scores
