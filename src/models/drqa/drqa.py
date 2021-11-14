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

        # context = [bs, ctx_len, emb_dim]
        # question = [bs, qtn_len, emb_dim]
        # question_mask = [bs, qtn_len]

        ctx_ = self.linear(context)
        ctx_ = self.relu(ctx_)
        # ctx_ = [bs, ctx_len, emb_dim]

        qtn_ = self.linear(question)
        qtn_ = self.relu(qtn_)
        # qtn_ = [bs, qtn_len, emb_dim]

        qtn_transpose = qtn_.permute(0, 2, 1)
        # qtn_transpose = [bs, emb_dim, qtn_len]

        align_scores = torch.bmm(ctx_, qtn_transpose)
        # align_scores = [bs, ctx_len, qtn_len]

        qtn_mask = question_mask.unsqueeze(1).expand(align_scores.size())
        # qtn_mask = [bs, 1, qtn_len] => [bs, ctx_len, qtn_len]

        # Fills elements of self tensor(align_scores) with value(-float(inf)) where mask is True. 
        # The shape of mask must be broadcastable with the shape of the underlying tensor.
        align_scores = align_scores.masked_fill(qtn_mask == 1, -float('inf'))
        # align_scores = [bs, ctx_len, qtn_len]

        align_scores_flat = align_scores.view(-1, question.size(1))
        # align_scores = [bs*ctx_len, qtn_len]

        alpha = F.softmax(align_scores_flat, dim=1)
        alpha = alpha.view(-1, context.shape[1], question.shape[1])
        # alpha = [bs, ctx_len, qtn_len]

        align_embedding = torch.bmm(alpha, question)
        # align = [bs, ctx_len, emb_dim]

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
        # x = [bs, seq_len, feature_dim]

        outputs = [x]
        for i in range(self.num_layers):

            lstm_input = outputs[-1]
            lstm_out = F.dropout(lstm_input, p=self.dropout)
            lstm_out, (hidden, cell) = self.lstms[i](lstm_input)

            outputs.append(lstm_out)

        output = torch.cat(outputs[1:], dim=2)
        # [bs, seq_len, num_layers*num_dir*hidden_dim]

        output = F.dropout(output, p=self.dropout)

        return output


class LinearAttentionLayer(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, question, question_mask):

        # question = [bs, qtn_len, input_dim] = [bs, qtn_len, bi_lstm_hid_dim]
        # question_mask = [bs,  qtn_len]

        qtn = question.view(-1, question.shape[-1])
        # qtn = [bs*qtn_len, hid_dim]

        attn_scores = self.linear(qtn)
        # attn_scores = [bs*qtn_len, 1]

        attn_scores = attn_scores.view(question.shape[0], question.shape[1])
        # attn_scores = [bs, qtn_len]

        attn_scores = attn_scores.masked_fill(question_mask == 1, -float('inf'))

        alpha = F.softmax(attn_scores, dim=1)
        # alpha = [bs, qtn_len]

        return alpha


def weighted_average(x, weights):
    # x = [bs, len, dim]
    # weights = [bs, len]

    weights = weights.unsqueeze(1)
    # weights = [bs, 1, len]

    w = weights.bmm(x).squeeze(1)
    # w = [bs, 1, dim] => [bs, dim]

    return w


class BilinearAttentionLayer(nn.Module):

    def __init__(self, context_dim, question_dim):

        super().__init__()

        self.linear = nn.Linear(question_dim, context_dim)

    def forward(self, context, question, context_mask):

        # context = [bs, ctx_len, ctx_hid_dim] = [bs, ctx_len, hid_dim*6] = [bs, ctx_len, 768]
        # question = [bs, qtn_hid_dim] = [bs, qtn_len, 768]
        # context_mask = [bs, ctx_len]

        qtn_proj = self.linear(question)
        # qtn_proj = [bs, ctx_hid_dim]

        qtn_proj = qtn_proj.unsqueeze(2)
        # qtn_proj = [bs, ctx_hid_dim, 1]

        scores = context.bmm(qtn_proj)
        # scores = [bs, ctx_len, 1]

        scores = scores.squeeze(2)
        # scores = [bs, ctx_len]

        scores = scores.masked_fill(context_mask == 1, -float('inf'))

        # alpha = F.log_softmax(scores, dim=1)
        # alpha = [bs, ctx_len]

        return scores


class DocumentReader(nn.Module):

    def __init__(self, hidden_dim, embedding_dim, num_layers, num_directions, dropout, device, glove_matrix_path):

        super().__init__()

        self.device_ = device

        # self.embedding = self.get_glove_embedding()

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

        # context = [bs, len_c]
        # question = [bs, len_q]
        # context_mask = [bs, len_c]
        # question_mask = [bs, len_q]

        ctx_embed = self.glove_embedding(context)
        # ctx_embed = [bs, len_c, emb_dim]

        ques_embed = self.glove_embedding(question)
        # ques_embed = [bs, len_q, emb_dim]

        ctx_embed = self.dropout(ctx_embed)

        ques_embed = self.dropout(ques_embed)

        align_embed = self.align_embedding(ctx_embed, ques_embed, question_mask)
        # align_embed = [bs, len_c, emb_dim]  

        ctx_bilstm_input = torch.cat([ctx_embed, align_embed], dim=2)
        # ctx_bilstm_input = [bs, len_c, emb_dim*2]

        ctx_outputs = self.context_bilstm(ctx_bilstm_input)
        # ctx_outputs = [bs, len_c, hid_dim*layers*dir] = [bs, len_c, hid_dim*6]

        qtn_outputs = self.question_bilstm(ques_embed)
        # qtn_outputs = [bs, len_q, hid_dim*6]

        qtn_weights = self.linear_attn_question(qtn_outputs, question_mask)
        # qtn_weights = [bs, len_q]

        qtn_weighted = weighted_average(qtn_outputs, qtn_weights)
        # qtn_weighted = [bs, hid_dim*6]

        start_scores = self.bilinear_attn_start(ctx_outputs, qtn_weighted, context_mask)
        # start_scores = [bs, len_c]

        end_scores = self.bilinear_attn_end(ctx_outputs, qtn_weighted, context_mask)
        # end_scores = [bs, len_c]

        return start_scores, end_scores
