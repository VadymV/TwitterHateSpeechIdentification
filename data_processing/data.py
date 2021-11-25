import random

import torch
from torchtext.legacy import data
from transformers import AutoTokenizer


class DatasetTwitterHateSpeech:
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base", normalization=True)

    def __init__(self, batch_size, model='cnn'):
        self.tweet = self.create_tweet_field(model)
        self.label = data.LabelField(dtype=torch.float)
        self.tweet_id = data.Field(dtype=torch.float, use_vocab=False, sequential=False)
        self.fields = [('id', self.tweet_id), ('l', self.label), ('t', self.tweet)]
        self.train_data, self.valid_data, self.test_data = self.create_splits()
        self.build_vocab(model)

        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            sort=False,
            batch_size=batch_size,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    @staticmethod
    def create_tweet_field(model):
        if model == "cnn":
            tweet = data.Field(tokenize='spacy',
                               tokenizer_language='en_core_web_sm',
                               batch_first=True)
        elif model == "roberta":
            init_token_idx = DatasetTwitterHateSpeech.tokenizer.cls_token_id
            eos_token_idx = DatasetTwitterHateSpeech.tokenizer.sep_token_id
            pad_token_idx = DatasetTwitterHateSpeech.tokenizer.pad_token_id
            unk_token_idx = DatasetTwitterHateSpeech.tokenizer.unk_token_id
            tweet = data.Field(batch_first=True,
                               use_vocab=False,
                               tokenize=DatasetTwitterHateSpeech.tokenize_and_cut,
                               preprocessing=DatasetTwitterHateSpeech.tokenizer.convert_tokens_to_ids,
                               init_token=init_token_idx,
                               eos_token=eos_token_idx,
                               pad_token=pad_token_idx,
                               unk_token=unk_token_idx)
        else:
            raise ValueError("An unknown model.")

        return tweet

    @staticmethod
    def tokenize_and_cut(tweet):
        tokens = DatasetTwitterHateSpeech.tokenizer.tokenize(tweet)
        tokens = tokens[
                 :512 - 2]  # The maximum length of tweets that can be handled (512 for BERT-based model); 2 additional tokens are reserved
        return tokens

    def create_splits(self):
        train_data, test_data = data.TabularDataset.splits(
            path='./raw_data',
            train='train_data.csv',
            test='test_data.csv',
            format='csv',
            fields=self.fields,
            skip_header=True
        )

        train_data, valid_data = train_data.split(random_state=random.getstate(),
                                                  split_ratio=0.8,
                                                  stratified=True,
                                                  strata_field='l')

        print(test_data.examples[0].id)
        print(train_data.examples[0].id)

        print("The number of training samples is {}".format(len(train_data)))
        print("The number of validation samples is {}".format(len(valid_data)))
        print("The number of test samples is {}".format(len(test_data)))

        pos, neg, ratio = DatasetTwitterHateSpeech.count_labels(train_data.examples)
        print("# labels in train data: pos={}, neg={}, ratio(neg/pos)={}".format(pos, neg, ratio))

        pos, neg, ratio = DatasetTwitterHateSpeech.count_labels(valid_data.examples)
        print("# labels in train data: pos={}, neg={}, ratio(neg/pos)={}".format(pos, neg, ratio))

        pos, neg, ratio = DatasetTwitterHateSpeech.count_labels(test_data.examples)
        print("# labels in train data: pos={}, neg={}, ratio(neg/pos)={}".format(pos, neg, ratio))

        return train_data, valid_data, test_data

    def build_vocab(self, model):
        if model == "cnn":
            self.tweet.build_vocab(self.train_data,
                                   max_size=25_000,
                                   vectors="glove.6B.100d",
                                   unk_init=torch.Tensor.normal_)

        self.label.build_vocab(self.train_data)
        print(self.label.vocab.stoi)

    @staticmethod
    def count_labels(data_):
        pos = 0
        neg = 0
        for example in data_:
            if example.l == "0":
                neg += 1
            elif example.l == "1":
                pos += 1
            else:
                raise ValueError("Wrong label")
        ratio = neg / pos

        return [pos, neg, ratio]
