import torch
from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules import weight_init
import numpy as np
from spotlight.layers import ScaledEmbedding, ZeroEmbedding

PADDING_IDX = 0


class SeqModel(nn.Module):
    def __init__(self, num_items, embedding_dim=64,
                 item_embedding_layer=None, sparse=False, nb_query=2):

        super(SeqModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_items = num_items
        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)
        lstm_hidden = 300
        bidirectional = False

        self.lstm = nn.GRU(batch_first=True,
                           input_size=embedding_dim,
                           hidden_size=lstm_hidden,
                           dropout=0.1,
                           bidirectional=bidirectional)
        self.lstm2projection = nn.Linear(lstm_hidden, num_items)
        weight_init(self.lstm2projection)
        # self.softmax = nn.LogSoftmax()

    def forward(self, item_sequences):
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)
        out, _ = self.lstm(sequence_embeddings)
        return self.lstm2projection(out)
