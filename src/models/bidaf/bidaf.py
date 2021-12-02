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


class CharacterEmbeddingLayer(nn.Module):

    def __init__(self, char_vocab_dim, char_emb_dim, num_output_channels, kernel_size):

        super().__init__()

        self.char_emb_dim = char_emb_dim
        self.char_embedding = nn.Embedding(char_vocab_dim, char_emb_dim, padding_idx=1)
        self.char_convolution = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.dropout(self.char_embedding(x))
        x = x.permute(0, 1, 3, 2)
        x = x.view(-1, self.char_emb_dim, x.shape[3])
        x = x.unsqueeze(1)
        x = self.relu(self.char_convolution(x))
        x = x.squeeze()
        x = F.max_pool1d(x, x.shape[2]).squeeze()
        x = x.view(batch_size, -1, x.shape[-1])

        return x   


class HighwayNetwork(nn.Module):

    def __init__(self, input_dim, num_layers=2):

        super().__init__()

        self.num_layers = num_layers

        self.flow_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])

    def forward(self, x):

        for i in range(self.num_layers):

            flow_value = F.relu(self.flow_layer[i](x))
            gate_value = torch.sigmoid(self.gate_layer[i](x))

            x = gate_value * flow_value + (1 - gate_value) * x

        return x


class ContextualEmbeddingLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.highway_net = HighwayNetwork(input_dim)

    def forward(self, x):
        highway_out = self.highway_net(x)
        outputs, _ = self.lstm(highway_out)
        return outputs


class BiDAF(nn.Module):

    def __init__(self, char_vocab_dim, emb_dim, char_emb_dim, num_output_channels, 
                 kernel_size, ctx_hidden_dim, device, glove_matrix_path):
        '''
        char_vocab_dim = len( char2idx ) 
        emb_dim = 100
        char_emb_dim =  8
        num_output_chanels = 100
        kernel_size = ( 8, 5 )
        ctx_hidden_dim = 100
        '''
        super().__init__()

        self.device = device

        self.word_embedding = self.get_glove_embedding(glove_matrix_path)
        self.character_embedding = CharacterEmbeddingLayer(char_vocab_dim, char_emb_dim, 
                                                           num_output_channels, kernel_size)
        self.contextual_embedding = ContextualEmbeddingLayer(emb_dim * 2, ctx_hidden_dim)
        self.dropout = nn.Dropout()
        self.similarity_weight = nn.Linear(emb_dim * 6, 1, bias=False)
        self.modeling_lstm = nn.LSTM(emb_dim * 8, emb_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.2)
        self.output_start = nn.Linear(emb_dim * 10, 1, bias=False)
        self.output_end = nn.Linear(emb_dim * 10, 1, bias=False)
        self.end_lstm = nn.LSTM(emb_dim * 2, emb_dim, bidirectional=True, batch_first=True)

    def get_glove_embedding(self, glove_matrix_path):

        weights_matrix = np.load(glove_matrix_path)
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device), freeze=True)

        return embedding

    def forward(self, ctx, ques, char_ctx, char_ques):
        ctx_len = ctx.shape[1]
        ques_len = ques.shape[1]

        ctx_word_embed = self.word_embedding(ctx)
        ques_word_embed = self.word_embedding(ques)
        ctx_char_embed = self.character_embedding(char_ctx)
        ques_char_embed = self.character_embedding(char_ques)

        # Concatenate word and character embeddings
        ctx_contextual_inp = torch.cat([ctx_word_embed, ctx_char_embed], dim=2)
        ques_contextual_inp = torch.cat([ques_word_embed, ques_char_embed], dim=2)
        ctx_contextual_emb = self.contextual_embedding(ctx_contextual_inp)
        ques_contextual_emb = self.contextual_embedding(ques_contextual_inp)

        # SIMILARITY
        ctx_ = ctx_contextual_emb.unsqueeze(2).repeat(1, 1, ques_len, 1)
        ques_ = ques_contextual_emb.unsqueeze(1).repeat(1, ctx_len, 1, 1)
        elementwise_prod = torch.mul(ctx_, ques_)
        alpha = torch.cat([ctx_, ques_, elementwise_prod], dim=3)
        similarity_matrix = self.similarity_weight(alpha).view(-1, ctx_len, ques_len)

        # C2Q

        a = F.softmax(similarity_matrix, dim=-1)        
        c2q = torch.bmm(a, ques_contextual_emb)

        # Q2C
        b = F.softmax(torch.max(similarity_matrix, 2)[0], dim=-1)
        b = b.unsqueeze(1)
        q2c = torch.bmm(b, ctx_contextual_emb)
        q2c = q2c.repeat(1, ctx_len, 1)

        G = torch.cat([ctx_contextual_emb, c2q, 
                       torch.mul(ctx_contextual_emb, c2q), 
                       torch.mul(ctx_contextual_emb, q2c)], dim=2)

        # Model
        M, _ = self.modeling_lstm(G)

        # Outp
        M2, _ = self.end_lstm(M)

        p1 = self.output_start(torch.cat([G, M], dim=2))
        p1 = p1.squeeze()

        # Pred

        p2 = self.output_end(torch.cat([G, M2], dim=2)).squeeze()

        return p1, p2
