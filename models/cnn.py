"""
The implementation of this model is taken from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb

Modifications:
 - Batch normalization is added.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))

        self.batch_norm_0 = nn.BatchNorm2d(n_filters)

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))

        self.batch_norm_1 = nn.BatchNorm2d(n_filters)

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tweet):

        # tweet = [batch size, tweet length]
        embedded = self.embedding(tweet)

        # embedded = [batch size, tweet length, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, tweet length, emb dim]

        conved_0 = F.relu(self.batch_norm_0(self.conv_0(embedded)).squeeze(3))
        conved_1 = F.relu(self.batch_norm_1(self.conv_1(embedded)).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        # conved_n = [batch size, n_filters, tweet length - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


