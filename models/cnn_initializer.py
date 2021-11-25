import torch
from torch import optim, nn

from data_processing.data import DatasetTwitterHateSpeech
from models.cnn import CNN
from utils import misc


class CNNInitializer:
    def __init__(self, batch_size, device, print_info=True):
        self.batch_size = batch_size
        self.print_info = print_info
        self.embedding_dim = 100
        self.num_filters = 100
        self.filter_sizes = [3, 4, 5]
        self.output_dim = 1
        self.dropout = 0.5

        self.dataset = DatasetTwitterHateSpeech(batch_size=batch_size, model='cnn')
        self.input_dim = len(self.dataset.tweet.vocab)

        self.model = self.create_model(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.IntTensor([13]))  # 29720(neg)/2242(pos)=~13
        self.criterion = self.criterion.to(device)

    def create_model(self, device):
        model = CNN(self.input_dim,
                    self.embedding_dim,
                    self.num_filters,
                    self.filter_sizes,
                    self.output_dim,
                    self.dropout,
                    self.dataset.tweet.vocab.stoi[self.dataset.tweet.pad_token])

        misc.count_parameters(model)

        pretrained_embeddings = self.dataset.tweet.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)

        unk_idx = self.dataset.tweet.vocab.stoi[self.dataset.tweet.unk_token]

        model.embedding.weight.data[unk_idx] = torch.zeros(self.embedding_dim)
        model.embedding.weight.data[self.dataset.tweet.vocab.stoi[self.dataset.tweet.pad_token]] = \
            torch.zeros(self.embedding_dim)

        model = model.to(device)
        if self.print_info:
            print(model)

        return model
