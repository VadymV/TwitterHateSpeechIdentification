import torch
from torch import optim, nn

from data_processing.data import DatasetTwitterHateSpeech
from models.roberta import RoBERTaBasedRNN


class RoBERTaInitializer:
    def __init__(self, batch_size, device, print_info=True):
        self.batch_size = batch_size
        self.print_model_and_params = print_info
        self.hidden_dim = 256
        self.num_layers = 2
        self.output_dim = 1
        self.bidirectional = True
        self.dropout = 0.25

        self.dataset = DatasetTwitterHateSpeech(batch_size=batch_size, model='roberta')

        self.model = self.create_model(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.IntTensor([13]))  # 29720(neg)/2242(pos)=~13
        self.criterion = self.criterion.to(device)

        self.freeze_bert()
        if self.print_model_and_params:
            self.count_parameters()

    def create_model(self, device):
        model = RoBERTaBasedRNN(
            self.hidden_dim,
            self.output_dim,
            self.num_layers,
            self.bidirectional,
            self.dropout)

        model = model.to(device)
        if self.print_model_and_params:
            print(model)

        return model

    def freeze_bert(self):
        for name, param in self.model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False

    def count_parameters(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('The model has {} trainable parameters'.format(trainable_params))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
