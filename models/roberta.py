"""
The implementation of this model is taken from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb

Modifications:
- A roBERTa-base model trained on ~58M tweets is used (https://huggingface.co/cardiffnlp/twitter-roberta-base)
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class RoBERTaBasedRNN(nn.Module):
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.roberta = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")

        embedding_dim = self.roberta.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tweet):

        # text = [batch size, tweet len]
        with torch.no_grad():
            embedded = self.roberta(tweet)[0]

        # embedded = [batch size, tweet len, emb dim]
        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        # hidden = [batch size, hid dim]

        output = self.out(hidden)
        # output = [batch size, out dim]

        return output
